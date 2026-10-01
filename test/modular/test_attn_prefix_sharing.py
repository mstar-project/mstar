"""Rows that share cached pages name more pages than the pool holds.

A captured-graph FlashInfer wrapper fixes its page-index buffer at construction,
and it was sized to the pool: without prefix reuse the rows of one step never
named a page twice, so they could not name more pages than exist. A hit hands the
same prefix pages to every row that matched it, and a decode step over rows that
all share a long prefix names each of those pages once per row. The plan then
overruns the buffer, though the rows occupy a fraction of the pool.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import (
    KVPlanOutput,
    KVPlanOutputs,
    PagedIndptrs,
    SequenceView,
)
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlashInfer plans on a device"
)

KV = "kv"
PAGE_SIZE = 16
POOL = 32
ROWS = 8
SHARED = 6  # pages every row names: ROWS * SHARED is past the pool


def _manager(max_seq_len: int = 4096) -> FlashInferManager:
    return FlashInferManager(
        kv_cache=KV, device=torch.device("cuda"), dtype=torch.bfloat16,
        kv_config=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=128, num_qo_heads=2,
            max_seq_len=max_seq_len, max_num_pages=POOL, page_size=PAGE_SIZE,
        ),
    )


def _shared_decode(rows: int, shared: int = SHARED) -> StepContext:
    """A leased decode step whose every row reads the same cached pages."""
    rids = tuple(f"r{i}" for i in range(rows))
    pages = list(range(1, shared + 1))
    views = [
        SequenceView(
            request_id=rid, label="main", page_idxs=pages,
            length=shared * PAGE_SIZE, to_compute=1,
        )
        for rid in rids
    ]
    indptrs = PagedIndptrs(
        qo_indptr=torch.arange(rows + 1, dtype=torch.int32),
        paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32) * shared,
        paged_kv_indices=torch.tensor(pages * rows, dtype=torch.int32),
        paged_kv_last_page_len=torch.full((rows,), PAGE_SIZE, dtype=torch.int32),
    )
    ctx = StepContext(
        request_ids=rids, graph_walk="decode", slot=0, capture=False,
        slot_lease=SlotLease(
            slot=0, bucket=BucketKey(graph_walk="decode", bs=rows, num_tokens=rows),
        ),
    )
    ctx.plan_results = {KV: KVPlanOutputs(
        {"main": KVPlanOutput(cpu_indptrs=indptrs, views=views)},
    )}
    return ctx


@requires_cuda
def test_rows_sharing_a_prefix_plan_past_the_pool_under_a_captured_graph():
    assert ROWS * SHARED > POOL, "the rows fit the pool, so this proves nothing"

    try:
        _manager().plan(AttentionStep(causal=True), _shared_decode(ROWS))
    except (RuntimeError, ValueError) as error:
        pytest.fail(
            f"{ROWS} rows sharing {SHARED} cached pages overran a page-index "
            f"buffer sized to the {POOL}-page pool: {error}"
        )


@requires_cuda
def test_one_row_holding_more_pages_than_its_positions_imply_still_plans():
    # Bagel writes a whole image at one position, so a row's KV is not bounded
    # by max_seq_len: a buffer sized from it would be smaller than the pool
    long_row = 20
    assert long_row * PAGE_SIZE > 64, "the row fits its positions, so this proves nothing"

    try:
        _manager(max_seq_len=64).plan(
            AttentionStep(causal=True), _shared_decode(1, shared=long_row),
        )
    except (RuntimeError, ValueError) as error:
        pytest.fail(
            f"a single row naming {long_row} of the pool's {POOL} pages overran "
            f"a buffer sized from the positions it spans: {error}"
        )
