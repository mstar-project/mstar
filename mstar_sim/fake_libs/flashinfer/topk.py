"""Stand-in for FlashInfer's DSA top-k page-table transform (``flashinfer.topk``).

Scores have no values on the emulated GPU, so each row "picks" its first ``min(n, k)`` pools in
order: the page-table slots (``page * page_size + offset``) of pools ``0 .. min(n, k) - 1``, -1
past them, and ``out_raw_indices`` (if given) their request-relative indices. The small-int page
table is real, so the slots are valid cache rows."""

from __future__ import annotations

import enum

import torch

from mstar_sim import clock


class TopKTieBreak(enum.IntEnum):
    NONE = 0
    SMALL = 1
    LARGE = 2


def _cost(scores, *args, **kwargs) -> tuple[float, float]:
    return 4.0 * scores.numel(), 2.0 * scores.numel() * scores.element_size()


@clock.kernel_fn("flashinfer.top_k_page_table_transform", cost=_cost)
def top_k_page_table_transform(scores, page_table, lengths, k, *, row_to_batch=None,
                               deterministic=False, tie_break=TopKTieBreak.NONE,
                               dsa_graph_safe=False, page_table_row_starts=None, page_size=1,
                               out_raw_indices=None):
    rows = scores.shape[0]
    device = page_table.device
    j = torch.arange(k, device=device, dtype=torch.long)[None, :]
    n = lengths.to(device=device, dtype=torch.long).reshape(-1)[:rows, None]
    live = j < torch.clamp(n, max=k)
    batch = (row_to_batch.to(device=device, dtype=torch.long).reshape(-1)[:rows]
             if row_to_batch is not None else torch.zeros(rows, dtype=torch.long, device=device))
    starts = (page_table_row_starts.to(device=device, dtype=torch.long).reshape(-1)[:rows, None]
              if page_table_row_starts is not None else torch.zeros(rows, 1, dtype=torch.long, device=device))
    col = torch.clamp(starts + j // page_size, max=page_table.shape[1] - 1)
    page = page_table.to(torch.long)[batch.clamp(max=page_table.shape[0] - 1)[:, None], col]
    slots = torch.where(live, page * page_size + j % page_size, -1).to(torch.int32)
    if out_raw_indices is not None:
        out_raw_indices.copy_(torch.where(live, j.expand(rows, k), -1).to(out_raw_indices.dtype))
    return slots
