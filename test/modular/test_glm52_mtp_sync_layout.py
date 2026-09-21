"""Row arithmetic for the PADDED MTP sync pass (glm52-gaps #1)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.glm52.submodules import mtp_sync_padded_layout  # noqa: E402


def test_single_request_full_acceptance():
    """e == k+1: nothing to pad, so this must match the eager layout exactly."""
    k = 2
    positions, last_rows, rewind = mtp_sync_padded_layout([3], [100], k)
    assert positions == [98, 99, 100]      # start-e+1 .. start
    assert last_rows == [2]                # last real row of the only request
    assert rewind == [0]                   # nothing padded, nothing to undo


def test_single_request_minimum_rows():
    """e == 1 (every draft rejected): 1 real row, k pads."""
    k = 2
    positions, last_rows, rewind = mtp_sync_padded_layout([1], [100], k)
    assert positions[0] == 100             # the one real token position
    assert positions == [100, 101, 102]    # pads continue monotonically
    assert last_rows == [0]                # draft 1 comes from row 0, not row 2
    assert rewind == [2]                   # undo the k pad advances


def test_pad_positions_never_repeat_or_go_backwards():
    k = 4
    for e in range(1, k + 2):
        positions, _, _ = mtp_sync_padded_layout([e], [50], k)
        assert len(positions) == k + 1
        assert positions == sorted(positions), (e, positions)
        assert len(set(positions)) == len(positions), (e, positions)
        assert min(positions) > 0


def test_real_rows_carry_exactly_the_eager_positions():
    """The real prefix must be byte-identical to what the eager path plans —
    padding may add rows, never move existing ones."""
    k = 3
    for e in range(1, k + 2):
        st = 77
        positions, last_rows, _ = mtp_sync_padded_layout([e], [st], k)
        assert positions[:e] == list(range(st - e + 1, st + 1))
        assert last_rows[0] == e - 1


def test_batched_requests_are_independent_blocks():
    """Each request owns a fixed rows=k+1 block, so last_rows is i*rows+e-1 —
    a cumsum (the eager formula) would be wrong here."""
    k = 2
    rows = k + 1
    e_list, starts = [1, 3, 2], [10, 20, 30]
    positions, last_rows, rewind = mtp_sync_padded_layout(e_list, starts, k)

    assert len(positions) == rows * len(e_list)
    assert last_rows == [0, 1 * rows + 2, 2 * rows + 1] == [0, 5, 7]
    assert rewind == [2, 0, 1]

    for i, (st, e) in enumerate(zip(starts, e_list, strict=True)):
        block = positions[i * rows:(i + 1) * rows]
        assert block[:e] == list(range(st - e + 1, st + 1))
        # the row draft 1 is taken from must carry the request's own `start`
        assert positions[last_rows[i]] == st


def test_rewind_restores_the_counter_for_every_e():
    """Runner advances `rows` per request; rewind must land back on `start`."""
    k = 5
    rows = k + 1
    for e in range(1, rows + 1):
        _, _, rewind = mtp_sync_padded_layout([e], [1000], k)
        assert rows - rewind[0] == e


def test_out_of_range_row_count_is_loud():
    k = 2
    with pytest.raises(ValueError, match=r"outside \[1, 3\]"):
        mtp_sync_padded_layout([4], [10], k)
    with pytest.raises(ValueError, match=r"outside \[1, 3\]"):
        mtp_sync_padded_layout([0], [10], k)


def test_mismatched_lengths_are_loud():
    with pytest.raises(ValueError):
        mtp_sync_padded_layout([1, 2], [10], 2)


def test_pack_padded_rows_matches_slice_assign():
    """The padded sync inputs (rows slots per request, real rows first,
    zeros after) built in two cats equal the per-request slice-assign."""
    import torch

    from mstar.model.glm52.submodules import mtp_pack_padded_rows

    torch.manual_seed(0)
    k = 3
    rows = k + 1
    for e_list in ([1], [4], [2, 4, 1], [3, 3, 3, 3], [1, 2, 3, 4, 2]):
        tokens = [torch.randint(0, 256, (e,)) for e in e_list]
        hiddens = [torch.randn(e, 6) for e in e_list]
        num = len(e_list)
        ref_ids = torch.zeros(num * rows, dtype=torch.long)
        ref_h = torch.zeros(num * rows, 6)
        for i, (t, h) in enumerate(zip(tokens, hiddens, strict=True)):
            ref_ids[i * rows:i * rows + t.shape[0]] = t
            ref_h[i * rows:i * rows + h.shape[0]] = h
        ids, h = mtp_pack_padded_rows(tokens, hiddens, rows)
        assert ids.dtype == torch.long and ids.shape == (num * rows,)
        assert h.shape == (num * rows, 6)
        assert torch.equal(ids, ref_ids), e_list
        assert torch.equal(h, ref_h), e_list
