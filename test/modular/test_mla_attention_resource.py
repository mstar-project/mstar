"""The MLA attention resource over a ``KVLayout.MLA`` KV resource, on CPU.

The FlashInfer kernel needs Hopper and the real dims, so these run the SDPA
fallback — over the same host plans, scatter maps and static buffers the
kernel path uses. Every step is checked against dense causal attention over
the tokens the request has seen, computed from the same latents.

Covers: the host plan math, prefill + decode, a batch of two at different
lengths across a page boundary, the speculative shape (k+1 verify rows,
rewind, a draft-phase step with k sub-plans over the same stream), and the
"kv length past the declared span" guard.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import Segment, StepContext, SubmoduleStep
from mstar.engine.resources.attn.mla import (
    MlaAttentionManager,
    MlaAttentionStep,
    MlaSubPlan,
    build_host_plan,
    paged_scatter_map_host,
)
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVLayout, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SequenceView
from mstar.engine.resources.runner import StepRunner

PAGE = 4
CKV, KPE, HEADS = 8, 4, 2
SCALE = (CKV + KPE) ** -0.5
DEV = torch.device("cpu")


class _StubTransferManager:
    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _make() -> tuple[KVManager, MlaAttentionManager, StepRunner]:
    cfg = KVConfig(
        num_layers=2, num_kv_heads=1, head_dim=CKV + KPE, max_seq_len=64,
        max_num_pages=32, page_size=PAGE, num_qo_heads=HEADS, layout=KVLayout.MLA,
    )
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=DEV, dtype=torch.float32,
    )
    attn = MlaAttentionManager(
        kv_cache="kv", device=DEV, dtype=torch.float32, kv_config=cfg,
        softmax_scale=SCALE, ckv_dim=CKV,
    )
    assert not attn.uses_kernel
    runner = StepRunner(
        {"kv": kv, "attn": attn}, node_resources={"n": ["kv", "attn"]},
    )
    return kv, attn, runner


def _ctx(rids) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False)


def _drive(runner, kv_step: KVStep, attn_step: MlaAttentionStep, rids, spans):
    step = SubmoduleStep(
        segments=[Segment(r, "main", s) for r, s in zip(rids, spans, strict=True)],
        steps={"kv": kv_step, "attn": attn_step},
    )
    step.set_ctx(_ctx(rids))
    assert runner.admit(step).ok
    runner.plan(step)
    return step


class _Reference:
    """Every latent a request has written, in order, for dense attention."""

    def __init__(self):
        self.rows: dict[str, list[torch.Tensor]] = {}

    def append(self, rid: str, latents: torch.Tensor):
        self.rows.setdefault(rid, []).extend(latents.unbind(0))

    def truncate(self, rid: str, n: int):
        self.rows[rid] = self.rows[rid][:n]

    def attend(self, rid: str, q: torch.Tensor, start: int) -> torch.Tensor:
        """``q`` [n, H, ckv+kpe] at absolute positions start..start+n-1."""
        keys = torch.stack(self.rows[rid][:start + q.shape[0]])  # (L, D)
        scores = torch.einsum("nhd,ld->nhl", q, keys) * SCALE
        pos = torch.arange(q.shape[0]) + start
        mask = torch.arange(keys.shape[0])[None, :] <= pos[:, None]
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        return torch.einsum("nhl,ld->nhd", scores.softmax(-1), keys[:, :CKV])


def _run_layer(attn, kv, latent, q, layer_idx=0):
    """What a layer does: write the pass's latents, then attend."""
    layer = kv.layer_view(layer_idx)
    attn.write_latent(latent, layer)
    return attn.run(q[..., :CKV], q[..., CKV:], layer)


def _rand(n):
    torch.manual_seed(n)
    latent = torch.randn(n, CKV + KPE)
    q = torch.randn(n, HEADS, CKV + KPE)
    return latent, q


# ── host plan math ──


def test_paged_scatter_map_host():
    # two requests: 3 new tokens at kv 5..7 over pages [10, 11]; 1 at kv 0 on [12]
    t2p, t2c = paged_scatter_map_host(
        qo_indptr=[0, 3, 4], kv_indptr=[0, 2, 3], kv_indices=[10, 11, 12],
        kv_len_arr=[8, 1], page_size=4,
    )
    assert t2p == [11, 11, 11, 12]
    assert t2c == [1, 2, 3, 0]


def test_build_host_plan_sub_plan_prefix_and_guard():
    views = [
        SequenceView("a", "main", page_idxs=[3, 4, 5], length=10, to_compute=4),
        SequenceView("b", "main", page_idxs=[7], length=2, to_compute=0),
    ]
    # attend a prefix: kv 7 of the 10 declared -> two pages
    plan = build_host_plan(views, MlaSubPlan(q_lens=(1, 0), kv_lens=(7, 2)), PAGE)
    assert plan.qo_indptr == [0, 1, 1]
    assert plan.kv_indptr == [0, 2, 3]
    assert plan.kv_indices == [3, 4, 7]
    assert plan.kv_len_arr == [7, 2]
    assert plan.page_tables == [[3, 4], [7]]
    with pytest.raises(ValueError, match="widen the segment"):
        build_host_plan(views, MlaSubPlan(q_lens=(1, 0), kv_lens=(11, 2)), PAGE)
    with pytest.raises(ValueError, match="requests for a step"):
        build_host_plan(views, MlaSubPlan(q_lens=(1,), kv_lens=(1,)), PAGE)


# ── the resource against dense attention ──


def test_prefill_then_decode_matches_dense():
    kv, attn, runner = _make()
    kv.ingest_request("a")
    ref = _Reference()

    # prefill 6 tokens (crosses a page)
    step = _drive(runner, KVStep(), MlaAttentionStep(), ["a"], [6])
    latent, q = _rand(6)
    out = _run_layer(attn, kv, latent, q)
    ref.append("a", latent)
    torch.testing.assert_close(out, ref.attend("a", q, 0), atol=1e-5, rtol=1e-5)
    runner.commit(step)
    assert kv.stored_len("a") == 6
    assert attn.qo_indptr_buf().tolist() == [0, 6]
    assert torch.equal(attn.select_last_hidden(torch.arange(6.0)[:, None]), torch.tensor([[5.0]]))

    # three decode steps
    for t in range(3):
        step = _drive(runner, KVStep(), MlaAttentionStep(), ["a"], [1])
        latent, q = _rand(100 + t)
        latent, q = latent[:1], q[:1]
        out = _run_layer(attn, kv, latent, q)
        ref.append("a", latent)
        torch.testing.assert_close(out, ref.attend("a", q, 6 + t), atol=1e-5, rtol=1e-5)
        runner.commit(step)
    assert kv.stored_len("a") == 9


def test_batch_of_two_packed_rows_and_second_layer():
    kv, attn, runner = _make()
    ref = _Reference()
    for rid, n in (("a", 5), ("b", 2)):
        kv.ingest_request(rid)
        step = _drive(runner, KVStep(), MlaAttentionStep(), [rid], [n])
        latent, q = _rand(n)
        _run_layer(attn, kv, latent, q, layer_idx=1)
        ref.append(rid, latent)
        runner.commit(step)

    # one packed step: a extends by 3, b by 2
    step = _drive(runner, KVStep(), MlaAttentionStep(), ["a", "b"], [3, 2])
    latent, q = _rand(5)
    out = _run_layer(attn, kv, latent, q, layer_idx=1)
    ref.append("a", latent[:3])
    ref.append("b", latent[3:])
    torch.testing.assert_close(out[:3], ref.attend("a", q[:3], 5), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[3:], ref.attend("b", q[3:], 2), atol=1e-5, rtol=1e-5)
    assert attn.page_tables() == [
        kv._streams["a"]["main"].page_indices[:2], kv._streams["b"]["main"].page_indices[:1]
    ]
    # layer 0 was never written in this test
    assert not kv.layer_view(0).any()
    runner.commit(step)


def test_speculative_verify_rewind_and_draft_phase_sub_plans():
    """The MTP shape at k=3: the trunk writes k+1 rows and commits them, the
    model keeps e, then one step runs the padded sync pass (k+1 rows at the
    rewound length) and k-1 chain rows, each its own sub-plan over the same
    stream, without committing."""
    k = 3
    kv, attn, runner = _make()
    kv.ingest_request("a")
    ref = _Reference()

    step = _drive(runner, KVStep(), MlaAttentionStep(), ["a"], [5])
    latent, q = _rand(5)
    _run_layer(attn, kv, latent, q)
    ref.append("a", latent)
    runner.commit(step)
    p0 = kv.stored_len("a")  # 5

    # trunk verify: k+1 rows at P0.., committed
    step = _drive(runner, KVStep(), MlaAttentionStep(), ["a"], [k + 1])
    latent, q = _rand(7)
    latent, q = latent[:k + 1], q[:k + 1]
    out = _run_layer(attn, kv, latent, q)
    ref.append("a", latent)
    torch.testing.assert_close(out, ref.attend("a", q, p0), atol=1e-5, rtol=1e-5)
    runner.commit(step)
    assert kv.stored_len("a") == p0 + k + 1

    # verify accepted e-1 drafts -> keep e rows
    e = 2
    kv.rewind("a", k + 1 - e)
    ref.truncate("a", p0 + e)
    assert kv.stored_len("a") == p0 + e

    # draft phase on the MTP plane (layer 1): sub-plan 0 = padded sync, k+1
    # rows at P0 attending P0+k+1; sub-plan it = one row at kv P0+e+it
    span = max(k + 1 - e, k - 1)
    sub_plans = (
        MlaSubPlan(q_lens=(k + 1,), kv_lens=(p0 + k + 1,)),
        *[MlaSubPlan(q_lens=(1,), kv_lens=(p0 + e + it,)) for it in range(1, k)],
    )
    step = _drive(
        runner, KVStep(commit=False), MlaAttentionStep(sub_plans=sub_plans), ["a"], [span],
    )
    plane = _Reference()
    plane.rows["a"] = [torch.zeros(CKV + KPE)] * p0  # the plane's prompt entries
    torch.manual_seed(7)
    kv.layer_view(1)[kv._streams["a"]["main"].page_indices[:2]] = 0.0

    attn.select_plan_slot(0)
    latent, q = _rand(k + 1)
    out = _run_layer(attn, kv, latent, q, layer_idx=1)
    plane.append("a", latent)
    torch.testing.assert_close(out, plane.attend("a", q, p0), atol=1e-5, rtol=1e-5)
    # rows >= e of the sync pass are rejected continuations; the chain
    # overwrites them from slot P0+e on
    plane.truncate("a", p0 + e)
    for it in range(1, k):
        attn.select_plan_slot(it)
        latent, q = _rand(200 + it)
        latent, q = latent[:1], q[:1]
        out = _run_layer(attn, kv, latent, q, layer_idx=1)
        plane.append("a", latent)
        torch.testing.assert_close(
            out, plane.attend("a", q, p0 + e + it - 1), atol=1e-5, rtol=1e-5,
        )
    attn.select_plan_slot(0)
    with pytest.raises(IndexError):
        attn.select_plan_slot(k)
    runner.commit(step)
    # nothing committed: the stream still holds P0 + e
    assert kv.stored_len("a") == p0 + e

    # the trunk (layer 0) was not touched by the plane's writes
    latent0 = kv.layer_view(0)
    written = [torch.stack(ref.rows["a"][:p0 + e])]
    pages = kv._streams["a"]["main"].page_indices
    flat = latent0[pages].reshape(-1, CKV + KPE)[:p0 + e]
    torch.testing.assert_close(flat, written[0])


def test_sub_plan_past_declared_span_is_refused():
    kv, attn, runner = _make()
    kv.ingest_request("a")
    step = _drive(runner, KVStep(), MlaAttentionStep(), ["a"], [3])
    runner.commit(step)
    with pytest.raises(ValueError, match="widen the segment"):
        _drive(
            runner, KVStep(commit=False),
            MlaAttentionStep(sub_plans=(MlaSubPlan(q_lens=(1,), kv_lens=(6,)),)),
            ["a"], [1],
        )
