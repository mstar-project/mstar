"""Speculative decoding: device-side registers for draft-and-verify steps.

A verify row carries k + 1 tokens ``[x_0, d_1 .. d_k]`` at positions ``P ..
P + k``: ``x_0`` is the last token the request emitted, the ``d_i`` the drafts,
``P`` the request's committed length. The target runs them as one causal
extend, and greedy acceptance keeps the longest run of drafts that equal the
target's own next token (``n`` of them), so the step emits ``n + 1`` tokens and
the stream grows by as many. The draft then runs ``k`` passes: pass 0 over the
verify's ``(rows, k + 1)`` tokens, fed the target's hidden state at ``P + i`` and
the target's token after it, which writes the draft's K/V for every position
the step may have kept; pass ``j > 0`` one token a row at ``P + n + j``, the
draft's own output from the pass before. Only slot ``n`` of pass 0 is a real
query, so every pass attends one query a row, over the verify's block table,
and its work after attention runs on ``rows`` rows. A pass's token is draft ``j
+ 1`` of the next step, written straight into that step's registers.

``P``, ``n`` and the next step's tokens live here, one register row per
request, and the host never reads them before the next step runs: the async
worker plans step N+1 while N is on the GPU, so N+1 is planned against an
optimistic length (all of N accepted) and corrected on the device. The KV write
slots, the attention lengths, the rotary positions and the recurrent state's
start slot all derive from ``P`` in-graph, through buffers the KV and attention
resources hand over in their plan results. The host learns ``n`` from the D2H
``check_stop`` already reads, and gives back what it over-counted in a later
commit (``KVStep.commit_adjust``).

What keeps the optimistic reservation sufficient: at every admit the host's
stream length is at least ``P + k - 1``. A verify step writes target KV at ``P ..
P + k`` and draft KV at up to ``P + 2k - 1``, and its view covers ``stored + k +
1 >= P + 2k``. The seed commit (the prompt's last prefill) adds ``k - 1`` past
the prompt to establish it, and each learned step gives back exactly what it
over-counted, ``(k + 1) - (n + 1)``.

A row also stops itself: tokens past a stop id or past the request's budget
are emitted as -1, and ``done`` tells the host, so a step never needs the host
to cut its output.

A sampled step (see `sampled`) keeps what greedy acceptance keeps, and also the
draft distribution each draft came from, here per request, until the next step
judges the draft against the target's.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.attn.prologue import QKV_NORM_ROPE_CACHE
from mstar.engine.resources.base import CGSlotKey, CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.sampler.unfiltered import SAMPLE_UNFILTERED
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.spec_decode import sampled
from mstar.engine.resources.spec_decode.config import (
    AcceptedTokens,
    DraftAddress,
    SpecDecodeSpec,
    SpecDecodeStep,
    SpecPhase,
)
from mstar.engine.resources.spec_decode.kernels import SPEC_ACCEPT, SPEC_VERIFY_INPUTS
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AllocationFailed,
    StepContext,
)
from mstar.utils.h2d import H2DMirror, PinnedStager

logger = logging.getLogger(__name__)

# The register row padding rows read and write; never handed to a request.
SINK_ROW = 0
# Fields of the per-step image staged in one H2D copy, each `rows` long.
_FIELDS = ("rows", "real", "base", "budget", "stop_on", "top_k")
# Where a request's sampled steps start in its Philox stream. The sampler resource
# counts from 0 for the prompt's first token, under the same seed, in the same word
# stream of the row; this is far enough up that the two cannot meet.
RNG_START = 1 << 31


class _RowSampling(NamedTuple):
    """This step's sampling settings, one entry per step row."""
    temperature: torch.Tensor
    top_k: torch.Tensor          # the vocabulary where the request's is off
    top_p: torch.Tensor
    seed: torch.Tensor
    draft_offset: torch.Tensor   # stream offset of the step's first draft draw


class _StepImage(NamedTuple):
    """One step's host-known per-row values on the device, as views of one
    int32 buffer filled by one copy, plus a buffer the step fills in-graph."""
    image: torch.Tensor          # [len(_FIELDS) * rows] int32
    rows: torch.Tensor           # register row per step row, SINK_ROW for padding
    real: torch.Tensor           # 1 for a real request, 0 for padding
    base: torch.Tensor           # seed steps: the stream's length after the prompt
    budget: torch.Tensor         # seed steps: decode tokens the request may emit
    stop_on: torch.Tensor        # seed steps: 0 where the request ignores stop ids
    top_k: torch.Tensor          # seed steps: the request's top_k (0: off)
    # seed steps: [temperature | top_p] per row, float32, its own copy
    params: torch.Tensor
    seeds: torch.Tensor          # seed steps: each request's sampling seed, int64
    num_accepted: torch.Tensor   # verify steps: the row's last accepted count, for the recurrent kernels
    # What an argmax through the unfiltered draw reads, [rows * (k + 1)]:
    # temperature 0, and offsets it stores back unchanged. Never read for a token.
    zero_temperature: torch.Tensor
    zero_seed: torch.Tensor
    zero_offset: torch.Tensor


class SpecDecodePlan(NamedTuple):
    """What ``plan`` hands its dependents: the phase, the rows, and the device
    buffers of the other resources a verify step rewrites per pass."""
    phase: SpecPhase
    num_rows: int
    image: _StepImage
    # attention's block table and per-row KV lengths, and KV's write slots
    block_table: torch.Tensor | None = None
    kv_lens: torch.Tensor | None = None
    write_pages: torch.Tensor | None = None
    write_offsets: torch.Tensor | None = None

    @property
    def num_accepted(self) -> torch.Tensor:
        """Each row's accepted count from its previous verify (1 after a seed),
        filled by ``verify_inputs`` before the layers run."""
        return self.image.num_accepted[: self.num_rows]


class SpecDecodeResource(Resource):
    @classmethod
    def build(cls, spec: SpecDecodeSpec, info: EngineResourceInfo) -> SpecDecodeResource:
        kv = info.dependency(spec.kv_cache)
        page_size = getattr(kv.config, "page_size", None)
        if page_size is None:
            raise ValueError(
                f"speculative decoding writes paged KV; {spec.kv_cache!r} is "
                f"{type(kv.config).__name__}"
            )
        return cls(
            device=info.device,
            num_draft_tokens=spec.config.num_draft_tokens,
            sampling=spec.config.sampling,
            vocab_size=spec.config.vocab_size,
            max_requests=spec.config.max_requests,
            page_size=page_size,
            kv_key=spec.kv_cache,
            attn_key=spec.attn,
            stop_token_ids=spec.stop_token_ids,
        )

    def __init__(
        self,
        device: torch.device,
        num_draft_tokens: int,
        max_requests: int,
        page_size: int,
        kv_key: str,
        attn_key: str,
        stop_token_ids: Sequence[int] = (),
        sampling: bool = False,
        vocab_size: int | None = None,
    ):
        self._device = device
        self.num_draft_tokens = num_draft_tokens
        self.tokens_per_row = num_draft_tokens + 1
        self._page_size = page_size
        self._kv_key = kv_key
        self._attn_key = attn_key
        rows = max_requests + 1
        i32 = dict(dtype=torch.int32, device=device)
        # Per request: committed length, last accepted count, the next verify's
        # tokens (by slot, so a draft pass's token master is a plain column),
        # decode tokens emitted, the budget, and whether stop ids apply.
        self._seq_len = torch.zeros(rows, **i32)
        self._num_acc = torch.ones(rows, **i32)
        self._tokens = torch.zeros(self.tokens_per_row, rows, dtype=torch.long, device=device)
        self._emitted = torch.zeros(rows, **i32)
        self._budget = torch.zeros(rows, **i32)
        self._stop_on = torch.zeros(rows, **i32)
        # per request, for a sampling node: temperature (0 greedy), top-p, top-k
        self.sampling = sampling
        self._temperature = torch.zeros(rows, dtype=torch.float32, device=device)
        self._top_p = torch.ones(rows, dtype=torch.float32, device=device)
        self._top_k = torch.zeros(rows, **i32)
        # the request's seed, and the Philox offset its next verify step draws at
        self._seed = torch.zeros(rows, dtype=torch.long, device=device)
        self._rng_offset = torch.zeros(rows, dtype=torch.long, device=device)
        self._rng_stride = sampled.words_per_step(num_draft_tokens)
        # per request and draft: the distribution the next verify judges it by
        self._draft_probs = torch.zeros(
            rows, num_draft_tokens, vocab_size, dtype=torch.float32, device=device,
        ) if sampling else None
        self._stop_ids = torch.tensor(
            sorted(set(stop_token_ids)) or [-2], dtype=torch.long, device=device,
        )
        self._zero_master = torch.zeros(rows, dtype=torch.long, device=device)

        self._lock = threading.Lock()
        # rid -> register row; reversed so `pop` hands out ascending rows
        self._row_of: dict[int, int] = {}
        self._free = list(reversed(range(1, rows)))
        # rid -> KV tokens over-counted by verify steps the host has learned
        # about but not yet given back
        self._retract: dict[int, int] = {}

        self._stager = PinnedStager(torch.int32)
        self._param_stager = PinnedStager(torch.float32)
        self._seed_stager = PinnedStager(torch.int64)
        self._cg_max_bs = 0
        self._cg_images: dict[CGSlotKey, tuple[_StepImage, H2DMirror]] = {}
        self._eager_images: dict[int, _StepImage] = {}

        self._current: SpecDecodePlan | None = None
        self._preplanned = False
        self._staged: SpecDecodePlan | None = None
        # set by `verify_inputs` / `accept_tokens` within one forward
        self._ids: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._slots: torch.Tensor | None = None
        self._keep: torch.Tensor | None = None
        self._row_sampling: _RowSampling | None = None
        # [k, rows]: each draft pass's KV slot, position and keys, computed
        # together once the step knows what it accepted
        self._draft_slots: torch.Tensor | None = None
        self._draft_positions: torch.Tensor | None = None
        self._draft_lens: torch.Tensor | None = None
        # bound by the engine; a draft pass writes and attends through them
        self._kv = None
        self._attn = None

    def depends_on(self) -> set[str]:
        # their plans are what this one rewrites on the device
        return {self._kv_key, self._attn_key}

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._kv = resources.get(self._kv_key)
        self._attn = resources.get(self._attn_key)

    # Request lifecycle

    def ingest_request(self, rid: int, overrides: ResourceReqConfig | None = None):
        with self._lock:
            if rid in self._row_of or not self._free:
                return
            self._row_of[rid] = self._free.pop()

    def ingest_capture_row(self, rid: int):
        # capture rows address the sink row
        return

    def remove_request(self, rid: int):
        with self._lock:
            row = self._row_of.pop(rid, None)
            self._retract.pop(rid, None)
            if row is not None:
                self._free.append(row)

    # Host bookkeeping

    def learn(self, request_ids: Sequence[int], accepted: Sequence[int]) -> None:
        """Record what verify steps accepted, read off the host copy
        ``check_stop`` already has. Each over-counted ``(k + 1) - accepted``
        KV tokens; the next step that declares this request gives them back."""
        full = self.tokens_per_row
        with self._lock:
            for rid, count in zip(request_ids, accepted, strict=True):
                if rid in self._row_of:
                    self._retract[rid] = self._retract.get(rid, 0) + full - int(count)

    def pending_retract(self, rid: int) -> int:
        """KV tokens a step declaring ``rid`` now may give back."""
        return self._retract.get(rid, 0)

    @property
    def seed_lookahead(self) -> int:
        """KV tokens a seed commit reserves past the prompt; see the module
        docstring for why a verify step needs them."""
        return self.num_draft_tokens - 1

    # Step lifecycle

    def admit(self, step: SpecDecodeStep, ctx: StepContext) -> AdmitOutcome:
        if ctx.capture:
            return ADMIT_OK
        for seg in step.segments or ():
            rid = seg.request_id
            if ctx.is_padding_row(rid) or rid in self._row_of:
                continue
            self.ingest_request(rid)
            if rid not in self._row_of:
                return AdmitOutcome(ok=False, reason=AllocationFailed(
                    message=(
                        f"speculative decoding holds {len(self._row_of)} requests, "
                        "its max_requests; none free"
                    ),
                    pages_short=1, label=seg.label, request_id=rid,
                ))
        return ADMIT_OK

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # a step's rows are staged into a slot's buffer behind an async H2D
        return True

    def clear_preplan(self):
        self._preplanned = False
        self._staged = None

    def plan(self, step: SpecDecodeStep, ctx: StepContext) -> SpecDecodePlan:
        if self._preplanned and not ctx.is_preplan:
            self._current = self._staged
            self.clear_preplan()
            return self._current
        segments = step.segments or ()
        num_rows = len(segments)
        image, mirror = self._image_for(ctx, num_rows)
        rows, real, base, budget, stop_on, top_k = (
            np.zeros(num_rows, dtype=np.int32) for _ in _FIELDS
        )
        temperature = np.zeros(num_rows, dtype=np.float32)
        top_p = np.ones(num_rows, dtype=np.float32)
        seeds = np.zeros(num_rows, dtype=np.int64)
        views = None
        if step.phase is SpecPhase.SEED:
            kv_plan = ctx.plan_results[self._kv_key].get("main")
            views = kv_plan.views if kv_plan is not None else None
        for i, seg in enumerate(segments):
            rid = seg.request_id
            if ctx.capture or ctx.is_padding_row(rid):
                continue
            rows[i] = self._row_of[rid]
            real[i] = 1
            if views is not None:
                base[i] = views[i].length
                budget[i] = step.budgets.get(rid, 0)
                stop_on[i] = rid not in step.ignore_stop
                if rid in step.sampling:
                    temperature[i], top_p[i], top_k[i], seeds[i] = step.sampling[rid]
        width = image.rows.numel()
        packed = np.zeros(len(_FIELDS) * width, dtype=np.int32)
        for f, values in enumerate((rows, real, base, budget, stop_on, top_k)):
            packed[f * width: f * width + num_rows] = values
        self._stager.copy_(image.image, packed, mirror=mirror)
        if self.sampling and step.phase is SpecPhase.SEED:
            params = np.ones(2 * width, dtype=np.float32)
            params[:num_rows] = temperature
            params[width: width + num_rows] = top_p
            self._param_stager.copy_(image.params, params)
            self._seed_stager.copy_(image.seeds, seeds, pad_value=0)

        plan = SpecDecodePlan(phase=step.phase, num_rows=num_rows, image=image)
        if step.phase is SpecPhase.VERIFY:
            plan = self._verify_plan(plan, ctx)
        if ctx.is_preplan:
            self._preplanned = True
            self._staged = plan
        else:
            self._current = plan
        return plan

    def _verify_plan(self, plan: SpecDecodePlan, ctx: StepContext) -> SpecDecodePlan:
        attn = (ctx.plan_results.get(self._attn_key) or {}).get("main")
        kv = (ctx.plan_results.get(self._kv_key) or {}).get("main")
        block_table = getattr(attn, "block_table", None)
        kv_lens = getattr(attn, "seqused_k", None)
        write_state = getattr(kv, "write_state", None)
        if block_table is None or kv_lens is None or write_state is None:
            raise RuntimeError(
                "speculative verify needs an attention backend whose plan reads "
                "per-row KV lengths and a block table on the device (AITER's "
                f"unified attention); {self._attn_key!r} planned {type(attn).__name__}"
            )
        n = plan.num_rows * self.tokens_per_row
        return plan._replace(
            block_table=block_table[: plan.num_rows],
            kv_lens=kv_lens[: plan.num_rows],
            write_pages=write_state.token_to_page[:n],
            write_offsets=write_state.token_to_cache[:n],
        )

    def commit(self, step: SpecDecodeStep, ctx: StepContext) -> None:
        if not step.retract:
            return
        with self._lock:
            for rid, given in step.retract.items():
                left = self._retract.get(rid, 0) - given
                if left > 0:
                    self._retract[rid] = left
                else:
                    self._retract.pop(rid, None)

    def _image_for(self, ctx: StepContext, num_rows: int) -> tuple[_StepImage, H2DMirror | None]:
        lease = ctx.slot_lease
        if lease is None:
            image = self._eager_images.get(ctx.slot)
            if image is None or image.rows.numel() < num_rows:
                image = self._eager_images[ctx.slot] = self._new_image(max(num_rows, 8))
            return image, None
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label="main")
        hit = self._cg_images.get(key)
        if hit is None:
            size = max(self._cg_max_bs, lease.bucket.bs, num_rows)
            hit = self._cg_images[key] = (self._new_image(size), H2DMirror())
        return hit

    def _new_image(self, rows: int) -> _StepImage:
        image = torch.zeros(len(_FIELDS) * rows, dtype=torch.int32, device=self._device)
        views = image.view(len(_FIELDS), rows)
        tokens = rows * self.tokens_per_row
        return _StepImage(
            image=image, rows=views[0], real=views[1], base=views[2],
            budget=views[3], stop_on=views[4], top_k=views[5],
            params=torch.ones(2 * rows, dtype=torch.float32, device=self._device),
            seeds=torch.zeros(rows, dtype=torch.long, device=self._device),
            num_accepted=torch.ones(rows, dtype=torch.int32, device=self._device),
            zero_temperature=torch.zeros(tokens, dtype=torch.float32, device=self._device),
            zero_seed=torch.zeros(tokens, dtype=torch.long, device=self._device),
            zero_offset=torch.zeros(tokens, dtype=torch.long, device=self._device),
        )

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._cg_images.clear()
        self._eager_images.clear()

    # In-graph operations, called by the forward in this order: `seed` on a
    # prompt's step; `verify_inputs`, `accept`, then `attend_draft` and
    # `record_draft` once per draft pass on a decode step.

    def _plan_now(self, phase: SpecPhase) -> SpecDecodePlan:
        plan = self._current
        if plan is None or plan.phase is not phase:
            raise RuntimeError(
                f"speculative decoding has no {phase.value} plan for this step; "
                "the step must declare a SpecDecodeStep of that phase"
            )
        return plan

    def _rows(self, plan: SpecDecodePlan) -> torch.Tensor:
        return plan.image.rows[: plan.num_rows].long()

    @torch.compiler.disable
    def seed(self, first_token: torch.Tensor, first_draft: torch.Tensor) -> None:
        """Start each prompt row's decode: its first token (sampled off the
        prompt), the draft head's guess after it, and its committed length."""
        plan = self._plan_now(SpecPhase.SEED)
        rows = self._rows(plan)
        b = plan.num_rows
        image = plan.image
        tokens = torch.cat(
            (first_token.view(1, b).long(),
             first_draft.view(1, b).long().expand(self.num_draft_tokens, b)),
            dim=0,
        )
        self._tokens.index_copy_(1, rows, tokens)
        self._seq_len.index_copy_(0, rows, image.base[:b])
        self._num_acc.index_fill_(0, rows, 1)
        self._emitted.index_fill_(0, rows, 0)
        self._budget.index_copy_(0, rows, image.budget[:b])
        self._stop_on.index_copy_(0, rows, image.stop_on[:b])
        if self.sampling:
            width = image.rows.numel()
            self._temperature.index_copy_(0, rows, image.params[:b])
            self._top_p.index_copy_(0, rows, image.params[width: width + b])
            self._top_k.index_copy_(0, rows, image.top_k[:b])
            self._seed.index_copy_(0, rows, image.seeds[:b])
            self._rng_offset.index_fill_(0, rows, RNG_START)
            # the first drafts are the head's argmax, a point mass each
            k, vocab = self.num_draft_tokens, self._draft_probs.shape[-1]
            point = torch.zeros(b, k, vocab, dtype=torch.float32, device=self._device)
            at = first_draft.view(b, 1, 1).long().expand(b, k, 1)
            self._draft_probs.index_copy_(0, rows, point.scatter_(2, at, sampled.DRAFT_MARGIN))

    @torch.compiler.disable
    def verify_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The verify tokens and their positions, ``[rows * (k + 1)]`` each;
        points this step's KV writes, attention lengths and recurrent start
        slots at the request's real length."""
        plan = self._plan_now(SpecPhase.VERIFY)
        image = plan.image
        b, n = plan.num_rows, plan.num_rows * self.tokens_per_row
        long = dict(dtype=torch.long, device=self._device)
        self._ids = torch.empty(n, **long)
        self._positions = torch.empty(n, **long)
        self._slots = torch.empty(n, **long)
        SPEC_VERIFY_INPUTS(
            image.rows[:b], image.real[:b], self._seq_len, self._num_acc, self._tokens,
            plan.block_table, self._page_size, self._ids, self._positions, self._slots,
            plan.write_pages, plan.write_offsets, plan.kv_lens, plan.num_accepted,
        )
        return self._ids, self._positions

    @torch.compiler.disable
    def accept(self, logits: torch.Tensor, sample: bool = False) -> AcceptedTokens:
        """Verification of this step's drafts against ``logits`` (``[rows *
        (k + 1), vocab]``, the target's at each verify position): acceptance
        while they equal its argmax, or with ``sample`` speculative sampling
        (see `sampled`) on each row's top-k/top-p distribution at its
        temperature, where a temperature of 0 is the argmax again."""
        if not sample:
            return self.accept_tokens(self._argmax(logits))
        if not self.sampling:
            raise RuntimeError("this speculative node was declared without sampling")
        plan = self._plan_now(SpecPhase.VERIFY)
        rows = self._rows(plan)
        b, t, vocab = plan.num_rows, self.tokens_per_row, logits.shape[-1]
        row = self._begin_sampling(rows, vocab)
        target = sampled.filtered_probs(
            logits, row.temperature.repeat_interleave(t), row.top_k.repeat_interleave(t),
            row.top_p.repeat_interleave(t),
        ).view(b, t, vocab)
        tokens, kept = sampled.accept_chain(
            self._draft_probs.index_select(0, rows), self._ids.view(b, t)[:, 1:], target,
            row.seed, row.draft_offset - t,
        )
        # -1 past the first rejection, which nothing emits but the draft embeds
        return self.accept_tokens(tokens.long().clamp(0, vocab - 1), kept=kept)

    def _argmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Each row's argmax, through the unfiltered draw's two kernels."""
        n, image = logits.shape[0], self._plan_now(SpecPhase.VERIFY).image
        out = torch.empty(n, dtype=torch.long, device=logits.device)
        SAMPLE_UNFILTERED(logits, image.zero_temperature[:n], image.zero_seed[:n], image.zero_offset[:n], out, 0)
        return out

    def _begin_sampling(self, rows: torch.Tensor, vocab: int) -> _RowSampling:
        """Read this step's settings and take its words of each request's
        stream: the accept chain's, then one for each draft draw."""
        top_k = self._top_k[rows]
        offset = self._rng_offset[rows]
        self._rng_offset.index_copy_(0, rows, offset + self._rng_stride)
        self._row_sampling = _RowSampling(
            temperature=self._temperature[rows],
            # 0 is off for the request, which the kernels would read as k = 0
            top_k=torch.where(top_k > 0, top_k, vocab),
            top_p=self._top_p[rows],
            seed=self._seed[rows],
            draft_offset=offset + self.tokens_per_row,
        )
        return self._row_sampling

    @torch.compiler.disable
    def accept_tokens(self, target: torch.Tensor, kept: torch.Tensor | None = None) -> AcceptedTokens:
        """Verification against the target's own token at each verify position
        (``[rows * (k + 1)]``), kept while it equals the next draft, or for
        ``kept`` drafts (``[rows]`` int32) when a sampler has decided that."""
        plan = self._plan_now(SpecPhase.VERIFY)
        image = plan.image
        b, t, k = plan.num_rows, self.tokens_per_row, self.num_draft_tokens
        device = self._device
        long = dict(dtype=torch.long, device=device)
        tokens = torch.empty(b, t, **long)
        accepted = torch.empty(b, dtype=torch.int32, device=device)
        done = torch.empty(b, dtype=torch.bool, device=device)
        self._keep = torch.empty(b, **long)
        self._draft_slots = torch.empty(k, b, **long)
        self._draft_positions = torch.empty(k, b, **long)
        self._draft_lens = torch.empty(k, b, dtype=torch.int32, device=device)
        SPEC_ACCEPT(
            image.rows[:b], self._ids, target, self._positions, plan.block_table, self._page_size,
            self._stop_ids, self._seq_len, self._num_acc, self._tokens, self._emitted, self._budget,
            self._stop_on, tokens, accepted, done, self._keep,
            self._draft_slots, self._draft_positions, self._draft_lens, kept,
        )
        return AcceptedTokens(tokens=tokens, accepted=accepted, done=done, target=target.view(b, t))

    def draft_address(self, j: int) -> DraftAddress:
        """Where draft pass ``j`` writes and reads: pass 0 writes every verify
        position, the others one position a row."""
        if j == 0:
            return DraftAddress(self._slots, self._positions, self._draft_lens[0])
        return DraftAddress(self._draft_slots[j], self._draft_positions[j], self._draft_lens[j])

    @torch.compiler.disable
    def draft_slots(self) -> torch.Tensor:
        """Each row's slot ``n``, the one real query of draft pass 0, as flat
        indices into ``[rows * (k + 1)]``."""
        return self._keep

    @torch.compiler.disable
    def attend_draft(
        self, j: int, qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor, heads: int, kv_heads: int, head_dim: int,
        eps: float, gated: bool, layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft pass ``j``'s attention over its packed ``[q | k | v]`` tokens:
        split, q/k norm, rotation (``cos``/``sin`` are tables over positions) and
        the KV write as one kernel, then one query a row. Returns the attention
        output and the gate, ``[rows, heads, head_dim]`` each."""
        where = self.draft_address(j)
        layer = self._kv.layer_view(layer_idx)
        q, gate, _, _ = QKV_NORM_ROPE_CACHE(
            qkv, q_weight, k_weight, cos, sin, where.positions, layer[:, 0], layer[:, 1], where.slots,
            heads, kv_heads, head_dim, eps, gated,
        )
        if j == 0:
            q = q.index_select(0, self._keep)
            if gated:
                gate = gate.index_select(0, self._keep)
        return self._attn.attend_rows(q, layer, where.kv_lens), gate

    @torch.compiler.disable
    def draw_draft(self, j: int, logits: torch.Tensor) -> torch.Tensor:
        """`record_draft` for a sampled step: pass ``j``'s token of each row, drawn
        from the draft's ``logits`` (``[rows, vocab]``) under the row's own
        sampling settings. Keeps the distribution for the next step to judge it by."""
        rows = self._rows(self._plan_now(SpecPhase.VERIFY))
        row = self._row_sampling
        probs = sampled.filtered_probs(logits, row.temperature, row.top_k, row.top_p)
        self._draft_probs[:, j].index_copy_(0, rows, probs * sampled.DRAFT_MARGIN)
        drafted = sampled.draw(probs, row.seed, row.draft_offset + j)
        self._tokens[j + 1].index_copy_(0, rows, drafted)
        return drafted

    @torch.compiler.disable
    def record_draft(self, j: int, logits: torch.Tensor) -> torch.Tensor:
        """Pass ``j``'s token of each row, the argmax of ``logits`` (``[rows,
        vocab]``): draft ``j + 1`` of the next step, written to its register by
        the same kernel, and the next pass's input."""
        plan = self._plan_now(SpecPhase.VERIFY)
        b, image = plan.num_rows, plan.image
        drafted = torch.empty(b, dtype=torch.long, device=logits.device)
        SAMPLE_UNFILTERED(
            logits, image.zero_temperature[:b], image.zero_seed[:b], image.zero_offset[:b], drafted, 0,
            image.rows[:b], self._zero_master, self._tokens[j + 1],
        )
        return drafted
