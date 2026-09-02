"""The draft-phase prepare/finish split (MSTAR_GLM52_MTP_PHASE_PREPARE).

The hoist moves the e-independent half of the draft phase above the verify
readback. Its whole safety argument is bookkeeping arithmetic, so it is
pinned here on CPU, where a violation is an assert instead of a silently
lower acceptance number on 8 GPUs:

1. the positions fed at prepare time (contiguous from P0) are IDENTICAL to
   what ``mtp_sync_padded_layout`` computes after the verify — not merely
   pad-equivalent;
2. prepare + finish issues the exact plan-call sequence the one-shot plan
   issues, and leaves the counters in the same end state;
3. prepare restores the live counters even when the slot-0 plan raises
   mid-way (the AllocationFailedError strand this lane's audit flagged).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.glm52.submodules import (  # noqa: E402
    Glm52LLMSubmodule,
    mtp_sync_padded_layout,
)

_MAIN = "main"


class _State:
    def __init__(self, seq_len: int, pos: int) -> None:
        self.seq_len = seq_len
        self.position_id_start = pos


class _FakeCM:
    """The exact surface _mtp_draft_phase_plan touches, with a call log."""

    def __init__(self, request_ids, seq_len_pos):
        self.request_ids = list(request_ids)
        self._states = {
            rid: _State(sl, pos)
            for rid, (sl, pos) in zip(request_ids, seq_len_pos, strict=True)
        }
        self.calls: list[tuple] = []
        self._last_planned: list[int] | None = None
        self.fail_on_plan_attention = False

    # -- surface -----------------------------------------------------------
    def set_active_label(self, label):
        pass

    def set_layer_idx(self, idx):
        pass

    def select_plan_slot(self, label, slot):
        self.calls.append(("slot", slot))

    def plan_attention(self, seq_lens, is_causal, label):
        if self.fail_on_plan_attention:
            raise RuntimeError("AllocationFailedError (simulated)")
        self._last_planned = list(seq_lens)
        self.calls.append(("attn", tuple(seq_lens)))

    def plan_rope(self, seq_lens, pos_ids, label):
        self.calls.append(("rope", tuple(pos_ids)))

    def advance_seq_lens(self, pos_id_ns=None):
        for rid, n in zip(self.request_ids, self._last_planned, strict=True):
            st = self._states[rid]
            st.seq_len += n
            st.position_id_start += n

    def rewind_seq_lens(self, ns):
        for rid, n in zip(self.request_ids, ns, strict=True):
            if n == 0:
                continue
            st = self._states[rid]
            if n < 0 or n > st.seq_len:
                raise ValueError(f"rewind {n} of {st.seq_len}")
            st.seq_len -= n
            st.position_id_start -= n

    def _get_state(self, rid, label=None):
        return self._states[rid]

    # -- helpers -----------------------------------------------------------
    def counters(self):
        return [
            (self._states[r].seq_len, self._states[r].position_id_start)
            for r in self.request_ids
        ]


def _sub(k: int) -> Glm52LLMSubmodule:
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = SimpleNamespace(mtp_num_draft_tokens=k, num_hidden_layers=78)
    return sub


def _shape(seq_lens):
    return SimpleNamespace(
        bs=len(seq_lens), seq_lens=list(seq_lens), total_tokens=sum(seq_lens))


def test_prepare_positions_match_padded_layout():
    """range(P0+1, P0+rows+1) == mtp_sync_padded_layout positions, always."""
    rng = random.Random(7)
    for _ in range(200):
        k = rng.randint(1, 4)
        rows = k + 1
        num = rng.randint(1, 5)
        e_list = [rng.randint(1, rows) for _ in range(num)]
        p0s = [rng.randint(0, 5000) for _ in range(num)]
        starts = [p0 + e for p0, e in zip(p0s, e_list, strict=True)]
        positions, last_rows, rewind = mtp_sync_padded_layout(e_list, starts, k)
        hoisted: list[int] = []
        for p0 in p0s:
            hoisted.extend(range(p0 + 1, p0 + 1 + rows))
        assert positions == hoisted
        assert last_rows == [i * rows + e - 1 for i, e in enumerate(e_list)]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("e_list", [[1], [2], None])
def test_phase_split_matches_one_shot_plan(k, e_list):
    """prepare(P0+rows) + finish(P0+e) == one-shot plan(P0), call for call."""
    rows = k + 1
    if e_list is None:
        e_list = [min(rows, 2), 1][:2]  # two requests
    e_list = [min(e, rows) for e in e_list]
    num = len(e_list)
    rids = [f"r{i}" for i in range(num)]
    p0s = [100 * (i + 1) for i in range(num)]
    sub = _sub(k)
    shape = _shape([rows] * num)

    # -- one-shot: the caller has rewound e; counters at P0 -----------------
    legacy = _FakeCM(rids, [(p0, p0) for p0 in p0s])
    sub._mtp_draft_phase_plan(legacy, shape, e_list=list(e_list))
    # ends at P0+e+(k-1), caller rewinds k-1 afterwards
    legacy.rewind_seq_lens([k - 1] * num)

    # -- split: prepare at P0+rows (post-trunk), finish at P0+e -------------
    split = _FakeCM(rids, [(p0 + rows, p0 + rows) for p0 in p0s])
    sub._mtp_draft_phase_plan(split, shape, phase="prepare")
    assert split.counters() == [(p0 + rows, p0 + rows) for p0 in p0s], (
        "prepare must leave the counters exactly where it found them")
    # simulate the verify: rewind rows-e (the m-e rewind in forward_batched)
    split.rewind_seq_lens([rows - e for e in e_list])
    sub._mtp_draft_phase_plan(split, shape, e_list=list(e_list), phase="finish")
    split.rewind_seq_lens([k - 1] * num)

    assert split.calls == legacy.calls, (
        "prepare+finish must issue the one-shot plan's exact call sequence")
    assert split.counters() == legacy.counters() == [
        (p0 + e, p0 + e) for p0, e in zip(p0s, e_list, strict=True)
    ], "both modes must end with the counters at P0+e"


def test_prepare_restores_counters_when_plan_raises():
    """A mid-plan failure must not strand the LIVE (aliased) counters."""
    k, rows = 3, 4
    rids = ["r0", "r1"]
    cm = _FakeCM(rids, [(104, 104), (204, 204)])
    cm.fail_on_plan_attention = True
    sub = _sub(k)
    with pytest.raises(RuntimeError, match="simulated"):
        sub._mtp_draft_phase_plan(cm, _shape([rows] * 2), phase="prepare")
    assert cm.counters() == [(104, 104), (204, 204)], (
        "the snapshot restore in the prepare phase must undo the rewind")


def test_finish_plans_only_chain_slots():
    k, rows = 3, 4
    rids = ["r0"]
    cm = _FakeCM(rids, [(102, 102)])  # P0=100, e=2
    cm._last_planned = [1]  # chain advance granularity
    _sub(k)._mtp_draft_phase_plan(cm, _shape([rows]), phase="finish")
    slots = [c[1] for c in cm.calls if c[0] == "slot"]
    attns = [c for c in cm.calls if c[0] == "attn"]
    assert slots == [1, 2, 0], "finish must touch slots 1..k-1 then reselect 0"
    assert all(a == ("attn", (1,)) for a in attns)
