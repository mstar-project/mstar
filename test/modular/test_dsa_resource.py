"""The DSA resource on CPU: its key store, the rows it plans from the KV plan, IndexShare's
shared pick, and sparse attention against an independent masked reference."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.resources.dsa.config import DSAConfig, DSASpec, DSAStep  # noqa: E402
from mstar.engine.resources.dsa.manager import (  # noqa: E402
    DSAManager,
    DSAPlan,
    IndexKeyStore,
    index_topk,
)
from mstar.engine.resources.kv.plan import SequenceView  # noqa: E402

# ---------------------------------------------------------------------------
# IndexKeyStore
# ---------------------------------------------------------------------------


def test_key_store_append_history_grow_and_evict():
    store = IndexKeyStore()
    d = 16
    first = torch.randn(3, d)
    second = torch.randn(70, d)  # crosses the initial 64-row capacity

    store.append("r0", 0, first, start_pos=0)
    store.append("r0", 0, second, start_pos=3)
    assert store.tokens("r0", 0) == 73
    hist = store.history("r0", 0, 73)
    torch.testing.assert_close(hist, torch.cat([first, second]), rtol=0, atol=0)
    # Windowed view (what a mid-sequence selection scores against).
    torch.testing.assert_close(store.history("r0", 0, 2), first[:2], rtol=0, atol=0)

    # Growth doubled capacity but content survived the buffer swap above;
    # a request/layer never touched reports 0, not KeyError.
    assert store.tokens("r0", 1) == 0
    assert store.tokens("nope", 0) == 0

    store.evict("r0")
    assert store.tracked_requests() == set()
    assert store.tokens("r0", 0) == 0
    store.evict("r0")  # idempotent


def test_key_store_desync_and_overrun_raise():
    store = IndexKeyStore()
    store.append("r0", 0, torch.randn(4, 8), start_pos=0)
    with pytest.raises(RuntimeError, match="desync"):
        store.append("r0", 0, torch.randn(1, 8), start_pos=3)  # overlap
    with pytest.raises(RuntimeError, match="desync"):
        store.append("r0", 0, torch.randn(1, 8), start_pos=5)  # gap
    with pytest.raises(RuntimeError, match="holds 4"):
        store.history("r0", 0, 5)  # window beyond stored history


def test_removing_a_request_drops_its_keys():
    dsa = _manager()
    dsa.keys.append("r0", 0, torch.randn(5, 16), start_pos=0)
    dsa.remove_request("r0")
    assert dsa.keys.tracked_requests() == set()
    dsa.remove_request("r0")  # idempotent, like the base hook


# ---------------------------------------------------------------------------
# plan: the rows of a step, from the KV plan
# ---------------------------------------------------------------------------


def _manager(topk=4, page_size=4) -> DSAManager:
    return DSAManager(DSAConfig(kv_cache="kv", topk=topk), page_size=page_size,
                      device=torch.device("cpu"))


def _plan(dsa: DSAManager, views: list[SequenceView]) -> dict[str, DSAPlan]:
    ctx = SimpleNamespace(plan_results={"kv": {"main": SimpleNamespace(views=views)}})
    return dsa.plan(DSAStep(), ctx)


def test_the_spec_builds_over_its_kv_cache():
    spec = DSASpec(resource_key="dsa", nodes={"LLM"}, config=DSAConfig(kv_cache="kv", topk=8))
    assert spec.depends_on() == {"kv"}
    assert spec.resource_class is DSAManager


def test_plan_reads_each_requests_rows_off_the_kv_plan():
    dsa = _manager(topk=16)
    tables = [[0, 1, 2], [5, 6, 7]]
    plans = _plan(dsa, [
        SequenceView("r0", "main", tables[0], length=4, to_compute=4),  # a 4-token prompt
        SequenceView("r1", "main", tables[1], length=9, to_compute=4),  # 5 cached, 4 new
    ])
    plan = plans["main"]
    assert plan.needs_selection is False  # every row sees at most topk keys: dense
    (s0, s1) = plan.spans
    assert (s0.request_id, s0.q_start, s0.q_len, s0.ctx_start) == ("r0", 0, 4, 0)
    assert (s1.request_id, s1.q_start, s1.q_len, s1.ctx_start) == ("r1", 4, 4, 5)
    assert s1.pages == [5, 6, 7]
    # Page tables are snapshots: later plan growth must not alias.
    tables[0].append(99)
    assert s0.pages == [0, 1, 2]


def test_plan_selects_once_a_row_sees_more_than_topk_keys():
    dsa = _manager(topk=4)
    _plan(dsa, [SequenceView("r0", "main", [0], length=4, to_compute=1)])
    assert dsa.needs_selection() is False
    _plan(dsa, [SequenceView("r0", "main", [0, 1], length=5, to_compute=1)])
    assert dsa.needs_selection() is True
    assert dsa._plans["main"].last_selection is None  # a fresh pick every forward


# ---------------------------------------------------------------------------
# select / shared_selection
# ---------------------------------------------------------------------------


def test_select_scores_each_request_over_its_own_keys():
    """Two decode rows of different requests each pick from their own keys, and SHARED layers
    get that very pick."""
    torch.manual_seed(0)
    topk, heads, d = 4, 2, 8
    dsa = _manager(topk=topk)
    keys = {"r0": torch.randn(6, d), "r1": torch.randn(9, d)}
    for rid, k in keys.items():
        dsa.keys.append(rid, 0, k[:-1], start_pos=0)
    _plan(dsa, [SequenceView("r0", "main", [0, 1], length=6, to_compute=1),
                SequenceView("r1", "main", [2, 3, 4], length=9, to_compute=1)])
    dsa.store_keys(0, torch.stack([keys["r0"][-1], keys["r1"][-1]]))
    q, w = torch.randn(2, heads, d), torch.rand(2, heads)

    picked = dsa.select(0, q, w)
    for row, k in enumerate(keys.values()):
        n = k.shape[0]
        want = index_topk(q[row:row + 1], w[row:row + 1], k, torch.tensor([n - 1]), topk)
        assert torch.equal(picked[row], want[0])
    assert dsa.shared_selection(1) is picked


def test_shared_layer_without_a_full_pick_raises():
    """A SHARED layer reached in selection mode with no published rows is a threading bug and
    must fail loudly, not attend densely off-spec."""
    dsa = _manager(topk=4)
    _plan(dsa, [SequenceView("r0", "main", [0, 1], length=7, to_compute=1)])
    with pytest.raises(RuntimeError, match="no FULL layer ran"):
        dsa.shared_selection(1)


@pytest.mark.parametrize("autocast", [False, True])
def test_scores_stay_fp32_under_bf16_autocast(autocast):
    torch.manual_seed(3)
    q, w, keys = torch.randn(3, 2, 8), torch.rand(3, 2), torch.randn(40, 8)
    positions = torch.tensor([20, 30, 39])
    ref = index_topk(q, w, keys, positions, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        assert torch.equal(index_topk(q, w, keys, positions, 8), ref)


# ---------------------------------------------------------------------------
# attend vs an independent mask-based reference
# ---------------------------------------------------------------------------


def _independent_masked_reference(latent_cache, pages, page_size, positions,
                                  selection_row, q_all, scale, latent_dim):
    """Dense MQA over ALL cached positions with an additive mask keeping only the selected
    ones — written from scratch, sharing no code with ``DSAManager.attend``."""
    pos = torch.arange(positions)
    flat = latent_cache[pages[pos // page_size], pos % page_size].float()
    scores = q_all.float() @ flat.T * scale  # (H, positions)
    mask = torch.full((positions,), float("-inf"))
    mask[selection_row[selection_row >= 0].long()] = 0.0
    weights = (scores + mask).softmax(-1)
    return weights @ flat[:, :latent_dim]


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("selection_row", [
    [9, 2, 7, 0],      # full topk row, unsorted, includes self (position 9)
    [5, 9, -1, -1],    # padded row: -1 entries must be excluded, not scored
])
def test_attend_matches_masked_reference(selection_row, autocast):
    """The engine runs the forward under bf16 autocast, which must not reach this fp32 math."""
    torch.manual_seed(2)
    page_size, heads, rank, rope = 4, 4, 32, 16
    dsa = _manager(topk=4, page_size=page_size)
    cache = torch.zeros(8, page_size, rank + rope)  # (pages, page_size, latent), the MLA layer
    pages = torch.tensor([3, 1, 6], dtype=torch.long)  # non-monotonic on purpose

    current = 9  # decode position; its own latent already written, as the layer does first
    for pos in range(current + 1):
        cache[pages[pos // page_size], pos % page_size] = torch.randn(rank + rope)
    _plan(dsa, [SequenceView("r0", "main", pages.tolist(), length=current + 1, to_compute=1)])

    q_nope = torch.randn(1, heads, rank)
    q_pe = torch.randn(1, heads, rope)
    selection = torch.tensor([selection_row], dtype=torch.int32)
    scale = (rank + rope) ** -0.5

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = dsa.attend(selection, q_nope, q_pe, cache, scale)

    q_all = torch.cat([q_nope, q_pe], dim=-1)[0]
    ref = _independent_masked_reference(
        cache, pages, page_size, current + 1, selection[0], q_all, scale, rank)
    torch.testing.assert_close(out[0], ref, rtol=1e-6, atol=1e-6)

    # Discrimination control: attending the WHOLE prefix must not match — otherwise this
    # test could not catch a sparse path that ignores the selection.
    dense_row = torch.arange(current + 1, dtype=torch.int32)
    dense_ref = _independent_masked_reference(
        cache, pages, page_size, current + 1, dense_row, q_all, scale, rank)
    assert not torch.allclose(out[0], dense_ref, rtol=1e-3, atol=1e-3)


def test_attend_batches_requests_over_their_own_pages():
    """Two decode rows of different requests, with their own page tables and a nonzero first
    slot, attend in one batch exactly as each would alone."""
    torch.manual_seed(4)
    page_size, heads, rank, rope = 4, 2, 16, 8
    dsa = _manager(topk=4, page_size=page_size)
    cache = torch.randn(10, page_size, rank + rope)
    views = [SequenceView("r0", "main", [7, 2], length=6, to_compute=1),
             SequenceView("r1", "main", [4, 9, 0], length=9, to_compute=1, start=2)]
    _plan(dsa, views)
    selection = torch.tensor([[5, 1, 3, -1], [8, 0, 6, 2]], dtype=torch.int32)
    q_nope, q_pe = torch.randn(2, heads, rank), torch.randn(2, heads, rope)
    out = dsa.attend(selection, q_nope, q_pe, cache, 0.25)

    for row, view in enumerate(views):
        pages = torch.tensor(view.page_idxs)
        slot = torch.arange(view.length) + view.start
        flat = cache[pages[slot // page_size], slot % page_size]
        ref = _independent_masked_reference(
            flat.unsqueeze(1), torch.arange(view.length), 1, view.length, selection[row],
            torch.cat([q_nope, q_pe], dim=-1)[row], 0.25, rank)
        torch.testing.assert_close(out[row], ref, rtol=1e-6, atol=1e-6)


def test_a_selecting_step_stages_its_page_tables_once(monkeypatch):
    """Every layer reads the same page tables: one H2D per step, not one per layer and row."""
    dsa = _manager(topk=4)
    calls = []
    stage = dsa._page_stager.to_device
    monkeypatch.setattr(dsa._page_stager, "to_device", lambda v, d: calls.append(v) or stage(v, d))
    _plan(dsa, [SequenceView("r0", "main", [0, 1], length=6, to_compute=1),
                SequenceView("r1", "main", [2, 3], length=7, to_compute=1)])
    for _ in range(3):
        dsa.attend(torch.zeros(2, 4, dtype=torch.int32), torch.randn(2, 2, 16),
                   torch.randn(2, 2, 8), torch.randn(4, 4, 24), 1.0)
    assert calls == [[0, 1, 2, 3, 0, 2, 0, 0]]  # pages, each row's offset, each row's start
    _plan(dsa, [SequenceView("r0", "main", [0], length=3, to_compute=1)])
    assert len(calls) == 1  # dense: nothing to stage


def test_plan_refuses_a_prefill_past_topk():
    dsa = _manager(topk=4)
    with pytest.raises(RuntimeError, match="decode-only"):
        _plan(dsa, [SequenceView("r0", "main", [1, 2], length=7, to_compute=2)])
