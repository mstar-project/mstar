"""A captured packed step's write addressing, computed on the host
(`_stage_packed_plan_state`), matches the device unpacking of the same
indptrs (`_compute_plan_state` into the captured state)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mstar.engine.resources.kv.manager import KVManager, KVPlanState
from mstar.engine.resources.kv.plan import SINK_PAGE, PagedIndptrs
from mstar.utils.h2d import H2DMirror, PinnedStager

PAGE, CAPTURE = 16, 96


def _indptrs(rng):
    """Prompt chunks, decode rows and zero-span padding rows on the sink."""
    qo, kvp, idx, last = [0], [0], [], []
    page = SINK_PAGE + 1
    while qo[-1] < CAPTURE - 20:
        new = int(rng.choice([1, 1, 1, int(rng.integers(2, 20))]))
        n = new + int(rng.integers(0, 3 * PAGE))
        pages = -(-n // PAGE)
        qo.append(qo[-1] + new)
        idx.extend(rng.permutation(np.arange(page, page + pages)).tolist())
        page += pages
        kvp.append(kvp[-1] + pages)
        last.append((n - 1) % PAGE + 1)
    for _ in range(3):
        qo.append(qo[-1])
        idx.append(SINK_PAGE)
        kvp.append(kvp[-1] + 1)
        last.append(1)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    return PagedIndptrs(
        qo_indptr=i32(qo), paged_kv_indptr=i32(kvp), paged_kv_indices=i32(idx),
        paged_kv_last_page_len=i32(last), kv_lens=None,
    )


def _state():
    """A captured state of CAPTURE tokens, as ``KVManager._static_plan_state``
    lays it out: pages then offsets in one buffer."""
    both = torch.full((2 * CAPTURE,), 7, dtype=torch.long)
    return KVPlanState(
        token_to_page=both[:CAPTURE], token_to_cache=both[CAPTURE:], staging=both, mirror=H2DMirror(),
    )


def _device_unpacked(kv, indptrs) -> KVPlanState:
    """The reference: the device unpacking, then copied into a captured state
    with SINK_PAGE / 0 past the real tokens."""
    total = int(indptrs.qo_indptr[-1])
    plan = kv._compute_plan_state(indptrs, total_tokens=total)
    ref = _state()
    ref.token_to_page[:total].copy_(plan.token_to_page)
    ref.token_to_cache[:total].copy_(plan.token_to_cache)
    ref.token_to_page[total:].fill_(SINK_PAGE)
    ref.token_to_cache[total:].fill_(0)
    ref.total_tokens = total
    return ref


def test_host_addressing_matches_device_unpacking():
    kv = KVManager.__new__(KVManager)
    kv.kv_cache = SimpleNamespace(page_size=PAGE)
    kv._device = torch.device("cpu")
    kv._plan_stager = PinnedStager(torch.long)
    rng = np.random.default_rng(0)
    for _ in range(20):
        indptrs = _indptrs(rng)
        ref = _device_unpacked(kv, indptrs)
        got = kv._stage_packed_plan_state(indptrs, _state(), CAPTURE)
        assert got.total_tokens == ref.total_tokens
        assert torch.equal(got.token_to_page[:CAPTURE], ref.token_to_page[:CAPTURE])
        assert torch.equal(got.token_to_cache[:CAPTURE], ref.token_to_cache[:CAPTURE])


def test_a_row_past_its_pages_is_refused():
    """A writing row with no page, or a last page longer than a page, would
    give a page index outside the row; numpy would wrap a negative one onto
    another row's page."""
    import pytest

    kv = KVManager.__new__(KVManager)
    kv.kv_cache = SimpleNamespace(page_size=PAGE)
    kv._plan_stager = PinnedStager(torch.long)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    for kvp, idx, last in (
        ([0, 1, 1], [5], [3, 1]),  # second row writes a token with no page
        ([0, 1, 2], [5, 6], [3, PAGE + 1]),  # second row's last page overflows
    ):
        indptrs = PagedIndptrs(
            qo_indptr=i32([0, 2, 3]), paged_kv_indptr=i32(kvp), paged_kv_indices=i32(idx),
            paged_kv_last_page_len=i32(last), kv_lens=None,
        )
        with pytest.raises(RuntimeError, match="past its own pages"):
            kv._stage_packed_plan_state(indptrs, _state(), CAPTURE)
