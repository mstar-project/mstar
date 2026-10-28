"""A packed step's write addressing, computed on the host from the plan's
indptrs (`_packed_addressing`), matches unpacking the same indptrs with torch
ops (the device path it replaced, kept here as the reference): staged into a
captured state (`_stage_packed_plan_state`) and uploaded for an eager write
(`_host_plan_state`). A step that writes nothing leaves a slot's buffer zero
and then issues no copy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mstar.engine.resources.kv.manager import KVManager, KVPlanState, _SlotBuffer
from mstar.engine.resources.kv.plan import SINK_PAGE, PagedIndptrs
from mstar.utils.h2d import PinnedStager

PAGE, CAPTURE = 16, 96


def _kv():
    kv = KVManager.__new__(KVManager)
    kv.kv_cache = SimpleNamespace(page_size=PAGE)
    kv._device = torch.device("cpu")
    kv._plan_stager = PinnedStager(torch.long)
    return kv


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


def _state(capture=CAPTURE, slot_buffer=None):
    """A captured state of ``capture`` tokens, as ``KVManager._static_plan_state``
    lays it out: pages then offsets at the front of the slot's buffer (by
    default a new one holding garbage, so not known zero)."""
    if slot_buffer is None:
        slot_buffer = _SlotBuffer(torch.full((2 * CAPTURE,), 7, dtype=torch.long))
        slot_buffer.zero_prefix = 0
    both = slot_buffer.buffer[:2 * capture]
    return KVPlanState(
        token_to_page=both[:capture], token_to_cache=both[capture:], staging=both, slot_buffer=slot_buffer,
    )


def _unpacked(indptrs, total):
    """The reference: each token's page and offset by torch ops on the indptrs."""
    qo, kvp = indptrs.qo_indptr.long(), indptrs.paged_kv_indptr.long()
    lens = qo[1:] - qo[:-1]
    kv_after = (kvp[1:] - kvp[:-1] - 1) * PAGE + indptrs.paged_kv_last_page_len.long()
    seg = torch.repeat_interleave(torch.arange(len(lens)), lens, output_size=total)
    g = (kv_after - lens)[seg] + torch.arange(total) - torch.repeat_interleave(qo[:-1], lens, output_size=total)
    page = torch.div(g, PAGE, rounding_mode="floor")
    return indptrs.paged_kv_indices.long()[kvp[:-1][seg] + page], g - page * PAGE


def _device_unpacked(kv, indptrs) -> KVPlanState:
    """The reference, copied into a captured state with SINK_PAGE / 0 past the
    real tokens."""
    total = int(indptrs.qo_indptr[-1])
    pages, offsets = _unpacked(indptrs, total)
    ref = _state()
    ref.token_to_page[:total].copy_(pages)
    ref.token_to_cache[:total].copy_(offsets)
    ref.token_to_page[total:].fill_(SINK_PAGE)
    ref.token_to_cache[total:].fill_(0)
    ref.total_tokens = total
    return ref


def test_host_addressing_matches_device_unpacking():
    kv = _kv()
    rng = np.random.default_rng(0)
    for _ in range(20):
        indptrs = _indptrs(rng)
        ref = _device_unpacked(kv, indptrs)
        got = kv._stage_packed_plan_state(indptrs, _state(), CAPTURE)
        assert got.total_tokens == ref.total_tokens
        assert torch.equal(got.token_to_page[:CAPTURE], ref.token_to_page[:CAPTURE])
        assert torch.equal(got.token_to_cache[:CAPTURE], ref.token_to_cache[:CAPTURE])


def test_a_step_that_only_reads_stages_the_sink():
    """Every row zero-span (a decoder reading the encoder context it wrote
    earlier): nothing is written, and every slot the graph scatters goes to
    SINK_PAGE, overwriting what the buffer held."""
    kv = _kv()
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    indptrs = PagedIndptrs(
        qo_indptr=i32([0, 0, 0, 0]), paged_kv_indptr=i32([0, 2, 3, 3]), paged_kv_indices=i32([5, 6, 9]),
        paged_kv_last_page_len=i32([4, PAGE, PAGE]), kv_lens=None,
    )
    got = kv._stage_packed_plan_state(indptrs, _state(), CAPTURE)
    assert got.total_tokens == 0
    assert got.token_to_page.tolist() == [SINK_PAGE] * CAPTURE
    assert got.token_to_cache.tolist() == [0] * CAPTURE


def test_a_row_past_its_pages_is_refused():
    """A writing row with no page, or a last page longer than a page, would
    give a page index outside the row; numpy would wrap a negative one onto
    another row's page."""
    import pytest

    kv = KVManager.__new__(KVManager)
    kv.kv_cache = SimpleNamespace(page_size=PAGE)
    kv._device = torch.device("cpu")
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


def test_an_eager_writes_host_addressing_matches_the_reference():
    kv = _kv()
    rng = np.random.default_rng(1)
    for _ in range(20):
        indptrs = _indptrs(rng)
        total = int(indptrs.qo_indptr[-1])
        pages, offsets = _unpacked(indptrs, total)
        got = kv._host_plan_state(indptrs, total)
        assert got.total_tokens == total
        assert torch.equal(got.token_to_page, pages)
        assert torch.equal(got.token_to_cache, offsets)


def test_a_zeroed_buffer_takes_no_copy_for_a_step_that_writes_nothing():
    """Once a zero-write step has zeroed a slot's buffer, another one at any
    capture length up to it is not staged; a write makes the next one copy."""
    kv = _kv()
    slot = _state().slot_buffer
    kv._stage_static(_state(CAPTURE, slot), (), (), 0)
    assert slot.buffer.tolist() == [0] * (2 * CAPTURE)
    # poke the buffer behind the stager's back: an untouched poke shows no copy ran
    slot.buffer[0] = 99
    kv._stage_static(_state(CAPTURE // 2, slot), (), (), 0)
    assert slot.buffer[0].item() == 99
    slot.buffer[0] = 0
    kv._stage_static(_state(4, slot), [5, 6], [1, 2], 2)
    assert slot.buffer[:8].tolist() == [5, 6, SINK_PAGE, SINK_PAGE, 1, 2, 0, 0]
    kv._stage_static(_state(4, slot), (), (), 0)
    assert slot.buffer[:8].tolist() == [0] * 8
