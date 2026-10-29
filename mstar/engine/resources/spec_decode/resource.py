"""Speculative decoding: device-side registers for draft-and-verify steps.

A verify row carries k + 1 tokens ``[x_0, d_1 .. d_k]`` at positions ``P ..
P + k``: ``x_0`` is the last token the request emitted, the ``d_i`` the drafts,
``P`` the request's committed length. The target runs them as one causal
extend, and greedy acceptance keeps the longest run of drafts that equal the
target's own next token (``n`` of them), so the step emits ``n + 1`` tokens and
the stream grows by as many. The draft then runs ``k`` passes over the same
``(rows, k + 1)`` shape: pass 0 at the verify positions, fed the target's hidden
state at ``P + i`` and the target's token after it; pass ``j > 0`` at ``P + j +
i``, where only slot ``n`` is a real query (the draft's own output from the
pass before, at the position after it). Slot ``n`` of pass ``j`` is draft ``j +
1`` of the next step. Every pass has the shape of the verify, so one capture
bucket, one attention plan and one KV write plan serve all of them.

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
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.spec_decode.config import (
    AcceptedTokens,
    SpecDecodeSpec,
    SpecDecodeStep,
    SpecPhase,
)
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
_FIELDS = ("rows", "real", "base", "budget", "stop_on")


class _StepImage(NamedTuple):
    """One step's host-known per-row values on the device, as views of one
    int32 buffer filled by one copy, plus a buffer the step fills in-graph."""
    image: torch.Tensor          # [len(_FIELDS) * rows] int32
    rows: torch.Tensor           # register row per step row, SINK_ROW for padding
    real: torch.Tensor           # 1 for a real request, 0 for padding
    base: torch.Tensor           # seed steps: the stream's length after the prompt
    budget: torch.Tensor         # seed steps: decode tokens the request may emit
    stop_on: torch.Tensor        # seed steps: 0 where the request ignores stop ids
    num_accepted: torch.Tensor   # verify steps: the row's last accepted count, for the recurrent kernels


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
        # tokens, decode tokens emitted, the budget, and whether stop ids apply.
        self._seq_len = torch.zeros(rows, **i32)
        self._num_acc = torch.ones(rows, **i32)
        self._tokens = torch.zeros(rows, self.tokens_per_row, dtype=torch.long, device=device)
        self._emitted = torch.zeros(rows, **i32)
        self._budget = torch.zeros(rows, **i32)
        self._stop_on = torch.zeros(rows, **i32)
        self._stop_ids = torch.tensor(
            sorted(set(stop_token_ids)) or [-2], dtype=torch.long, device=device,
        )
        self._slot_offsets = torch.arange(self.tokens_per_row, **i32)

        self._lock = threading.Lock()
        # rid -> register row; reversed so `pop` hands out ascending rows
        self._row_of: dict[int, int] = {}
        self._free = list(reversed(range(1, rows)))
        # rid -> KV tokens over-counted by verify steps the host has learned
        # about but not yet given back
        self._retract: dict[int, int] = {}

        self._stager = PinnedStager(torch.int32)
        self._cg_max_bs = 0
        self._cg_images: dict[CGSlotKey, tuple[_StepImage, H2DMirror]] = {}
        self._eager_images: dict[int, _StepImage] = {}

        self._current: SpecDecodePlan | None = None
        self._preplanned = False
        self._staged: SpecDecodePlan | None = None
        # set by `verify_inputs` / `accept` within one forward
        self._ids: torch.Tensor | None = None
        self._base: torch.Tensor | None = None
        self._accepted: torch.Tensor | None = None

    def depends_on(self) -> set[str]:
        # their plans are what this one rewrites on the device
        return {self._kv_key, self._attn_key}

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
        rows, real, base, budget, stop_on = (
            np.zeros(num_rows, dtype=np.int32) for _ in _FIELDS
        )
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
        width = image.rows.numel()
        packed = np.zeros(len(_FIELDS) * width, dtype=np.int32)
        for f, values in enumerate((rows, real, base, budget, stop_on)):
            packed[f * width: f * width + num_rows] = values
        self._stager.copy_(image.image, packed, mirror=mirror)

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
        return _StepImage(
            image=image, rows=views[0], real=views[1], base=views[2],
            budget=views[3], stop_on=views[4],
            num_accepted=torch.ones(rows, dtype=torch.int32, device=self._device),
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
    # prompt's step; `verify_inputs`, `accept`, then `draft_positions` /
    # `record_draft` once per draft pass on a decode step.

    def _plan_now(self, phase: SpecPhase) -> SpecDecodePlan:
        plan = self._current
        if plan is None or plan.phase is not phase:
            raise RuntimeError(
                f"speculative decoding has no {phase.value} plan for this step; "
                "the step must declare a SpecDecodeStep of that phase"
            )
        return plan

    def _rows(self, plan: SpecDecodePlan) -> tuple[torch.Tensor, torch.Tensor]:
        image = plan.image
        return image.rows[: plan.num_rows].long(), image.real[: plan.num_rows].bool()

    @torch.compiler.disable
    def seed(self, first_token: torch.Tensor, first_draft: torch.Tensor) -> None:
        """Start each prompt row's decode: its first token (sampled off the
        prompt), the draft head's guess after it, and its committed length."""
        plan = self._plan_now(SpecPhase.SEED)
        rows, _ = self._rows(plan)
        b = plan.num_rows
        image = plan.image
        tokens = torch.cat(
            (first_token.view(b, 1).long(),
             first_draft.view(b, 1).long().expand(b, self.num_draft_tokens)),
            dim=1,
        )
        self._tokens.index_copy_(0, rows, tokens)
        self._seq_len.index_copy_(0, rows, image.base[:b])
        self._num_acc.index_fill_(0, rows, 1)
        self._emitted.index_fill_(0, rows, 0)
        self._budget.index_copy_(0, rows, image.budget[:b])
        self._stop_on.index_copy_(0, rows, image.stop_on[:b])

    @torch.compiler.disable
    def verify_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The verify tokens and their positions, ``[rows * (k + 1)]`` each;
        points this step's KV writes, attention lengths and recurrent start
        slots at the request's real length."""
        plan = self._plan_now(SpecPhase.VERIFY)
        rows, real = self._rows(plan)
        base = torch.where(real, self._seq_len[rows], 0)
        plan.num_accepted.copy_(torch.where(real, self._num_acc[rows], 1))
        self._base = base
        positions = base[:, None] + self._slot_offsets[None, :]
        self._address(plan, positions, None, base + self.tokens_per_row)
        self._ids = self._tokens[rows].reshape(-1)
        return self._ids, positions.reshape(-1)

    @torch.compiler.disable
    def accept(self, logits: torch.Tensor) -> AcceptedTokens:
        """Greedy verification of this step's drafts against ``logits``
        (``[rows * (k + 1), vocab]``, the target's at each verify position)."""
        return self.accept_tokens(logits.argmax(dim=-1))

    @torch.compiler.disable
    def accept_tokens(self, target: torch.Tensor) -> AcceptedTokens:
        """Verification against the target's own token at each verify position
        (``[rows * (k + 1)]``): its argmax, or what a sampler drew there."""
        plan = self._plan_now(SpecPhase.VERIFY)
        rows, _ = self._rows(plan)
        b, t = plan.num_rows, self.tokens_per_row
        y = target.view(b, t).long()
        x = self._ids.view(b, t)
        accepted = (x[:, 1:] == y[:, :-1]).int().cumprod(dim=1).sum(dim=1) + 1
        self._accepted = accepted
        self._seq_len.index_copy_(0, rows, (self._base + accepted).int())
        self._num_acc.index_copy_(0, rows, accepted.int())
        last = y.gather(1, (accepted - 1)[:, None].long())
        self._tokens[:, :1].index_copy_(0, rows, last)

        slots = self._slot_offsets[None, :]
        emit = slots < accepted[:, None]
        stops = torch.isin(y, self._stop_ids) & emit & (self._stop_on[rows] > 0)[:, None]
        hit = stops.any(dim=1)
        first_stop = torch.where(hit, stops.int().argmax(dim=1), t)
        emit &= slots <= first_stop[:, None]
        emitted = self._emitted[rows]
        emit &= slots < (self._budget[rows] - emitted)[:, None]
        emitted = emitted + emit.sum(dim=1, dtype=torch.int32)
        self._emitted.index_copy_(0, rows, emitted)
        return AcceptedTokens(
            tokens=torch.where(emit, y, -1),
            accepted=accepted.int(),
            done=hit | (emitted >= self._budget[rows]),
            target=y,
        )

    @torch.compiler.disable
    def draft_positions(self, j: int) -> torch.Tensor:
        """Positions of draft pass ``j`` (``[rows * (k + 1)]``), and its KV
        writes and attention lengths: pass 0 at the verify positions, writing
        every slot; pass ``j > 0`` one further on, writing only slot ``n``."""
        plan = self._plan_now(SpecPhase.VERIFY)
        base, t = self._base, self.tokens_per_row
        positions = base[:, None] + j + self._slot_offsets[None, :]
        mask = None
        if j > 0:
            mask = self._slot_offsets[None, :] == (self._accepted - 1)[:, None]
        self._address(plan, positions, mask, base + t + j)
        return positions.reshape(-1)

    @torch.compiler.disable
    def record_draft(self, j: int, draft: torch.Tensor) -> torch.Tensor:
        """Keep slot ``n`` of pass ``j``'s tokens (``[rows * (k + 1)]``) as
        draft ``j + 1``; returns it in every slot, as pass ``j + 1``'s input."""
        plan = self._plan_now(SpecPhase.VERIFY)
        rows, _ = self._rows(plan)
        b, t = plan.num_rows, self.tokens_per_row
        chosen = draft.view(b, t).long().gather(1, (self._accepted - 1)[:, None].long())
        self._tokens[:, j + 1: j + 2].index_copy_(0, rows, chosen)
        return chosen.expand(b, t).reshape(-1)

    def _address(
        self, plan: SpecDecodePlan, positions: torch.Tensor,
        mask: torch.Tensor | None, kv_lens: torch.Tensor,
    ) -> None:
        """Point this pass's KV writes at ``positions`` (masked slots at the
        sink page) and its attention at ``kv_lens`` keys per row."""
        page = torch.div(positions, self._page_size, rounding_mode="floor")
        pages = plan.block_table.gather(1, page.long()).long()
        offsets = (positions - page * self._page_size).long()
        if mask is not None:
            pages = torch.where(mask, pages, SINK_PAGE)
            offsets = torch.where(mask, offsets, 0)
        plan.write_pages.copy_(pages.reshape(-1))
        plan.write_offsets.copy_(offsets.reshape(-1))
        plan.kv_lens.copy_(kv_lens)
