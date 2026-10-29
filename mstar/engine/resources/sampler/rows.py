"""The sampler's share of a persistent walk's rows (``mstar.engine.rows``).

Each row holds its request's settings (temperature, top-k, ...), mirrored
from the host masters, and its RNG offset, last token and declared loop-back
states, checked out of the slot masters; with a repetition penalty, also its
seen-token mask, checked out of the request's own. The walk's sampler reads
and advances them in place, so a step that changes no rows gathers and
scatters nothing; the masters are only read when a row joins and written when
it leaves.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from mstar.engine.resources.sampler.config import LoopbackState
from mstar.engine.resources.sampler.utils import (
    AcceleratorGraphableSampler,
    FilterOrder,
    SamplerBuffers,
)
from mstar.engine.rows import RowSpace, RowState, StepRows


class SamplerRows(RowState):
    def __init__(
        self, space: RowSpace, buffers: Callable[[], SamplerBuffers],
        loopback_state: Mapping[str, LoopbackState],
        seen_masks: tuple[int, Callable[[int], torch.Tensor | None]] | None = None,
    ):
        """``seen_masks``: the vocabulary width and each request's seen-token
        mask (None before its vocabulary is known), for a penalised sampler."""
        # looked up per use: the buffers are reallocated until capture is done
        self._buffers = buffers
        bufs = buffers()
        self.scalars = {
            name: space.buffer(f"sampler.{name}", (), buf.dtype, pad=buf.default)
            for name, buf in bufs.scalar_fields().items()
        }
        self.offset = space.buffer("sampler.offset", (), torch.long, mirrored=False)
        self.last_token = space.buffer("sampler.last_token", (), torch.long, mirrored=False)
        self.loopback = {
            name: space.buffer(
                f"sampler.{name}", (state.width,), state.dtype, pad=state.default, mirrored=False,
            )
            for name, state in loopback_state.items()
        }
        # never read on the device; kept in the rows so it moves with them
        self.master_slot = space.buffer("sampler.master_slot", (), torch.long)
        self.seen = None
        if seen_masks is not None:
            width, self._seen_mask = seen_masks
            self.seen = space.buffer("sampler.seen", (width,), torch.bool, pad=False, mirrored=False)
        self._samplers: dict[int, AcceleratorGraphableSampler] = {}

    def check_out(self, rows: StepRows) -> None:
        d = rows.diff
        bufs = self._buffers()
        masters = bufs.check_out_slots(d.join_rids)
        with bufs._host_master_lock:
            for name, row in self.scalars.items():
                rows.write(row, d.join_slots, getattr(bufs, name).master[masters])
        rows.write(self.master_slot, d.join_slots, masters)
        rows.copy_in(self.offset, d.join_slots, bufs.offset.master, masters)
        rows.copy_in(self.last_token, d.join_slots, bufs.last_token.master, masters)
        for name, row in self.loopback.items():
            rows.copy_in(row, d.join_slots, bufs.loopback[name].master, masters)
        if self.seen is not None:
            self._check_out_seen(rows)
        rows.on_commit(lambda released: bufs.hold_slots(masters))

    def _check_out_seen(self, rows: StepRows) -> None:
        d = rows.diff
        masks = [self._seen_mask(rid) for rid in d.join_rids]
        have = [i for i, m in enumerate(masks) if m is not None]
        if have:
            rows.copy_in_each(self.seen, d.join_slots[have], [masks[i] for i in have])
        if len(have) < len(masks):
            blank = [i for i, m in enumerate(masks) if m is None]
            rows.write(self.seen, d.join_slots[blank], np.zeros((len(blank), *self.seen.shape), np.bool_))

    def check_in(self, rows: StepRows) -> None:
        d = rows.diff
        bufs = self._buffers()
        masters = self.master_slot.host[d.leave_slots]
        keep = [i for i, rid in enumerate(d.leave_rids) if rid not in d.dropped]
        if keep:
            slots, dst = d.leave_slots[keep], masters[keep]
            rows.copy_out(self.offset, slots, bufs.offset.master, dst)
            rows.copy_out(self.last_token, slots, bufs.last_token.master, dst)
            for name, row in self.loopback.items():
                rows.copy_out(row, slots, bufs.loopback[name].master, dst)
            if self.seen is not None:
                masks = [self._seen_mask(d.leave_rids[i]) for i in keep]
                back = [i for i, m in enumerate(masks) if m is not None]
                if back:
                    rows.copy_out_each(self.seen, slots[back], [masks[i] for i in back])
        # every leaving row, removed or not: the slot of a removed request is
        # freed only now, after its check-in was queued
        rows.on_commit(lambda released: bufs.drop_slots(masters))

    def sampler_for(self, bs: int) -> AcceleratorGraphableSampler:
        """A sampler over the first ``bs`` rows, cached: the rows never move."""
        sampler = self._samplers.get(bs)
        if sampler is None:
            def view(name: str) -> torch.Tensor | None:
                row = self.scalars.get(name)
                return None if row is None else row.rows(bs)

            order = None
            if "top_p_first" in self.scalars:
                order = FilterOrder(view("top_p_first"), view("top_p_min_keep"))
            sampler = self._samplers[bs] = AcceleratorGraphableSampler(
                temperature_buf=view("temperature"),
                top_k_buf=view("top_k"),
                top_p_buf=view("top_p"),
                seed_buf=view("seed"),
                offset_buf=self.offset.rows(bs),
                rep_penalty_buf=view("rep_penalty"),
                min_p_buf=view("min_p"),
                order=order,
                last_token_buf=self.last_token.rows(bs),
                seen_tokens_buf=None if self.seen is None else self.seen.rows(bs),
                tp_group=self._buffers().tp_group,
            )
        sampler.applied_penalty_in_graph = False
        return sampler
