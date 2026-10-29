"""The KV resource's share of a persistent walk's rows (``mstar.engine.rows``).

A row holds, per label the walk's steps extend by one token (``RowWalk.labels``),
its stream's stored length, mirrored and stepped by each forward, and its page
list, on the device only. A step then builds its write addressing and
FlashInfer's index arrays on the device from the rows (``decode_addressing``,
``decode_indptrs``), inside its graph, and the host plans from the mirrored
lengths alone; what the host uploads is only a joining row, or a page a row
crossed onto.

The streams stay the host truth: a step's commit still extends each one, and
pages are only ever appended while a request holds a row (forks, offload and
the like check the row in first), so the rows need nothing read back.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.rows import RowFills, RowSpace, StepRows

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: the torch fills below serve them
    triton = None


class KVDeviceRows(NamedTuple):
    """One label's rows on the device."""
    lens: torch.Tensor  # (max_rows,) int32: stored length, before the step's token
    pages: torch.Tensor  # (max_rows, max_pages) int32, SINK_PAGE past the stream's
    page_size: int


if triton is not None:
    @triton.jit
    def _decode_addressing_kernel(lens, pages, page_out, offset_out, n, max_pages, page_size, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        live = r < n
        length = tl.load(lens + r, mask=live, other=0)
        at = length // page_size
        tl.store(page_out + r, tl.load(pages + r * max_pages + at, mask=live), mask=live)
        tl.store(offset_out + r, length - at * page_size, mask=live)

    @triton.jit
    def _decode_indptrs_kernel(
        lens, pages, indptr_out, indices_out, last_page_len_out, n, max_pages, page_size, capacity,
        BLOCK: tl.constexpr, PAGES: tl.constexpr,
    ):
        r = tl.arange(0, BLOCK)
        live = r < n
        length = tl.load(lens + r, mask=live, other=0)
        counts = tl.where(live, length // page_size + 1, 0)
        ends = tl.cumsum(counts, 0)
        tl.store(indptr_out, 0)
        tl.store(indptr_out + 1 + r, ends, mask=live)
        tl.store(last_page_len_out + r, length - (counts - 1) * page_size + 1, mask=live)
        starts = ends - counts
        for first in range(0, max_pages, PAGES):
            col = first + tl.arange(0, PAGES)
            dst = starts[:, None] + col[None, :]
            held = live[:, None] & (col[None, :] < counts[:, None]) & (dst < capacity)
            page = tl.load(pages + r[:, None] * max_pages + col[None, :], mask=held)
            tl.store(indices_out + dst, page, mask=held)


def _on_triton(rows: KVDeviceRows) -> bool:
    return triton is not None and rows.lens.device.type == "cuda"


def decode_addressing(rows: KVDeviceRows, n: int, page_out: torch.Tensor, offset_out: torch.Tensor) -> None:
    """Where each of the first ``n`` rows writes its token: the page holding
    position ``lens`` and the slot in it. A padding row (length 0, sink pages)
    writes SINK_PAGE."""
    if _on_triton(rows):
        _decode_addressing_kernel[(1,)](
            rows.lens, rows.pages, page_out, offset_out, n, rows.pages.shape[1], rows.page_size,
            BLOCK=triton.next_power_of_2(n),
        )
        return
    lens = rows.lens[:n].long()
    at = torch.div(lens, rows.page_size, rounding_mode="floor")
    page_out[:n].copy_(rows.pages[:n].gather(1, at.unsqueeze(1)).squeeze(1))
    offset_out[:n].copy_(lens - at * rows.page_size)


def decode_indptrs(
    rows: KVDeviceRows, n: int, indptr_out: torch.Tensor, indices_out: torch.Tensor,
    last_page_len_out: torch.Tensor,
) -> None:
    """FlashInfer's paged-KV arrays for the first ``n`` rows, each holding
    its stored length plus the step's token: the pages that covers, in row
    order, in the first ``indptr[n]`` entries of ``indices_out`` (as far as it
    holds), which are all FlashInfer reads."""
    if _on_triton(rows):
        _decode_indptrs_kernel[(1,)](
            rows.lens, rows.pages, indptr_out, indices_out, last_page_len_out, n,
            rows.pages.shape[1], rows.page_size, indices_out.shape[0],
            BLOCK=triton.next_power_of_2(n), PAGES=min(triton.next_power_of_2(rows.pages.shape[1]), 32),
        )
        return
    page_size = rows.page_size
    max_pages = rows.pages.shape[1]
    lens = rows.lens[:n].long()
    counts = torch.div(lens, page_size, rounding_mode="floor") + 1
    ends = counts.cumsum(0)
    indptr_out[:1].zero_()
    indptr_out[1:n + 1].copy_(ends)
    last_page_len_out[:n].copy_(lens - (counts - 1) * page_size + 1)
    m = min(indices_out.shape[0], n * max_pages)
    j = torch.arange(m, device=lens.device)
    row = torch.searchsorted(ends, j, right=True).clamp_(max=n - 1)
    col = (j - (ends - counts)[row]).clamp_(0, max_pages - 1)
    indices_out[:m].copy_(rows.pages[:n].reshape(-1)[row * max_pages + col])


class KVRows(RowFills):
    def __init__(self, space: RowSpace, labels: Iterable[str], max_pages: int, page_size: int):
        super().__init__()
        self.page_size = page_size
        self.max_pages = max_pages
        self.labels = {
            label: (
                space.buffer(f"kv.{label}.lens", (), torch.int32, step=1),
                space.buffer(f"kv.{label}.pages", (max_pages,), torch.int32, pad=SINK_PAGE, mirrored=False),
            )
            for label in labels
        }

    def device_rows(self, label: str) -> KVDeviceRows:
        lens, pages = self.labels[label]
        return KVDeviceRows(lens.device, pages.device, self.page_size)

    def check_out(self, rows: StepRows) -> None:
        # nothing yet: a joining row is written at plan (`write_joins`), once
        # admit has given its stream the page this step writes into
        return

    def write_joins(self, rows: StepRows, streams: dict) -> None:
        """The joining rows' lengths and pages, from their streams."""
        d = rows.diff
        out = np.full((len(d.join_rids), self.max_pages), SINK_PAGE, np.int32)
        for label, (lens, pages) in self.labels.items():
            stored = np.zeros(len(d.join_rids), np.int32)
            out.fill(SINK_PAGE)
            for i, rid in enumerate(d.join_rids):
                stream = streams[rid][label]
                stored[i] = stream.stored_len
                held = stream.page_indices
                if held.n > self.max_pages:
                    raise ValueError(
                        f"{rid}/{label} holds {held.n} pages, more than a row's {self.max_pages}"
                    )
                out[i, :held.n] = held.array()
            rows.write(lens, d.join_slots, stored)
            rows.write(pages, d.join_slots, out)

    def write_pages(self, rows: StepRows, label: str, slot: int, first: int, new_pages) -> None:
        """Pages a held row's stream gained this step, from page ``first`` on."""
        n = len(new_pages)
        if first + n > self.max_pages:
            raise ValueError(f"a row of {label} grows past its {self.max_pages} pages")
        rows.write_at(
            self.labels[label][1], np.full(n, slot), (np.arange(first, first + n),),
            np.asarray(new_pages, np.int32),
        )

    def lens_host(self, rows: StepRows, label: str) -> np.ndarray:
        """This step's lengths, before its token, by slot."""
        return rows.mirror(self.labels[label][0])
