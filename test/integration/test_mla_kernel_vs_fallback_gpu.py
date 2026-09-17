"""The MLA attention resource on the FlashInfer kernel (real dims, Hopper)
against its own SDPA fallback over the same host plans, on one GPU.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import Segment, StepContext, SubmoduleStep  # noqa: E402
from mstar.engine.resources.attn.mla import (  # noqa: E402
    MlaAttentionManager,
    MlaAttentionStep,
    MlaSubPlan,
    SdpaMLAWrapper,
    _mla_kernel_available,
    build_host_plan,
)
from mstar.engine.resources.kv.config import KVConfig, KVLayout, KVStep  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CKV, KPE, HEADS, PAGE = 512, 64, 8, 16
SCALE = (192 + KPE) ** -0.5


def _kernel_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    return _mla_kernel_available(CKV, KPE, torch.cuda.get_device_capability(0)[0])


def _make(device):
    from mstar.communication.tensors import LocalTransferEngine

    cfg = KVConfig(
        num_layers=2, num_kv_heads=1, head_dim=CKV + KPE, max_seq_len=256,
        max_num_pages=64, page_size=PAGE, num_qo_heads=HEADS, layout=KVLayout.MLA,
    )
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, device=device, dtype=torch.bfloat16,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="t", my_session_id="s", transfer_engine=LocalTransferEngine("localhost"),
        ),
    )
    attn = MlaAttentionManager(
        kv_cache="kv", device=device, dtype=torch.bfloat16, kv_config=cfg,
        softmax_scale=SCALE, ckv_dim=CKV,
    )
    runner = StepRunner({"kv": kv, "attn": attn}, node_resources={"n": ["kv", "attn"]})
    return kv, attn, runner


def _drive(runner, rids, spans, sub_plans=None, commit=True):
    step = SubmoduleStep(
        segments=[Segment(r, "main", s) for r, s in zip(rids, spans, strict=True)],
        steps={"kv": KVStep(commit=commit), "attn": MlaAttentionStep(sub_plans=sub_plans)},
    )
    step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    return step


class _Fallback:
    """The SDPA wrapper planned over the resource's own host plans, on a
    private copy of the latent layer, so the two paths share pages."""

    def __init__(self, attn, kv, device):
        self.attn, self.kv, self.device = attn, kv, device
        self.w = None

    def run(self, latent, q, layer_idx, sub=0):
        host = self.attn._plan_for(None).host_plans[sub]
        n = len(host.kv_len_arr)
        self.w = SdpaMLAWrapper(
            num_heads=HEADS, head_dim_ckv=CKV, page_size=PAGE, sm_scale=SCALE,
            device=self.device, batch_size=n, max_pages_per_request=32,
            max_total_tokens=max(host.total_tokens, 1),
        )
        self.w.plan(host, causal=True, dtype=torch.bfloat16)
        layer = self.kv.layer_view(layer_idx).clone()
        self.w.write_latent(layer, latent)
        out = self.w.run(q[..., :CKV], q[..., CKV:], layer)
        return out, layer


def _kernel(attn, kv, latent, q, layer_idx, sub=0):
    attn.select_plan_slot(sub)
    layer = kv.layer_view(layer_idx)
    attn.write_latent(latent, layer)
    return attn.run(q[..., :CKV], q[..., CKV:], layer)


def _rand(n, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(n, CKV + KPE, generator=g).to(device, torch.bfloat16)
    q = torch.randn(n, HEADS, CKV + KPE, generator=g).to(device, torch.bfloat16)
    return latent, q


def _close(a, b, what):
    a, b = a.float(), b.float()
    err = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=2e-2, rtol=2e-2), f"{what}: max abs err {err:.4f}"


@pytest.mark.skipif(not _kernel_ok(), reason="FlashInfer MLA kernel needs sm90 + flashinfer.mla")
def test_kernel_matches_fallback_prefill_decode_and_speculative_shape():
    device = torch.device("cuda")
    kv, attn, runner = _make(device)
    fb = _Fallback(attn, kv, device)
    kv.ingest_request("a")

    # prefill 37 tokens (three pages)
    step = _drive(runner, ["a"], [37])
    latent, q = _rand(37, device, 1)
    ref, _ = fb.run(latent, q, 0)
    out = _kernel(attn, kv, latent, q, 0)
    _close(out, ref, "prefill")
    runner.commit(step)

    # decode x3
    for t in range(3):
        step = _drive(runner, ["a"], [1])
        latent, q = _rand(1, device, 10 + t)
        ref, _ = fb.run(latent, q, 0)
        out = _kernel(attn, kv, latent, q, 0)
        _close(out, ref, f"decode {t}")
        runner.commit(step)
    assert kv.stored_len("a") == 40

    # speculative: k=3 verify rows, keep e=2, draft phase with 3 sub-plans
    k = 3
    step = _drive(runner, ["a"], [k + 1])
    latent, q = _rand(k + 1, device, 20)
    ref, _ = fb.run(latent, q, 0)
    out = _kernel(attn, kv, latent, q, 0)
    _close(out, ref, "verify")
    runner.commit(step)
    e = 2
    kv.rewind("a", k + 1 - e)
    p0 = kv.stored_len("a") - e
    sub_plans = (
        MlaSubPlan(q_lens=(k + 1,), kv_lens=(p0 + k + 1,)),
        *[MlaSubPlan(q_lens=(1,), kv_lens=(p0 + e + it,)) for it in range(1, k)],
    )
    step = _drive(
        runner, ["a"], [max(k + 1 - e, k - 1)], sub_plans=sub_plans, commit=False,
    )
    latent, q = _rand(k + 1, device, 30)
    ref, _ = fb.run(latent, q, 1, sub=0)
    out = _kernel(attn, kv, latent, q, 1, sub=0)
    _close(out, ref, "sync sub-plan")
    for it in range(1, k):
        latent, q = _rand(1, device, 40 + it)
        ref, _ = fb.run(latent, q, 1, sub=it)
        out = _kernel(attn, kv, latent, q, 1, sub=it)
        _close(out, ref, f"chain sub-plan {it}")
    runner.commit(step)
    assert kv.stored_len("a") == p0 + e


@pytest.mark.skipif(not _kernel_ok(), reason="FlashInfer MLA kernel needs sm90 + flashinfer.mla")
def test_kernel_matches_fallback_packed_batch():
    device = torch.device("cuda")
    kv, attn, runner = _make(device)
    fb = _Fallback(attn, kv, device)
    for rid, n in (("a", 21), ("b", 5)):
        kv.ingest_request(rid)
        step = _drive(runner, [rid], [n])
        latent, q = _rand(n, device, 50 + n)
        _kernel(attn, kv, latent, q, 0)
        runner.commit(step)
    step = _drive(runner, ["a", "b"], [4, 2])
    latent, q = _rand(6, device, 60)
    ref, _ = fb.run(latent, q, 0)
    out = _kernel(attn, kv, latent, q, 0)
    _close(out, ref, "packed batch")
    runner.commit(step)


def test_host_plan_sub_plan_math_on_device_views():
    """No kernel needed: the host plan of a sub-plan is what both paths read."""
    device = torch.device("cuda")
    kv, attn, runner = _make(device)
    kv.ingest_request("a")
    step = _drive(runner, ["a"], [20])
    runner.commit(step)
    step = _drive(
        runner, ["a"], [3],
        sub_plans=(MlaSubPlan(q_lens=(2,), kv_lens=(22,)), MlaSubPlan(q_lens=(1,), kv_lens=(18,))),
        commit=False,
    )
    views = step.ctx.plan_results["kv"]["main"].views
    h0 = build_host_plan(views, MlaSubPlan((2,), (22,)), PAGE)
    h1 = build_host_plan(views, MlaSubPlan((1,), (18,)), PAGE)
    assert h0.kv_len_arr == [22] and len(h0.kv_indices) == 2
    assert h1.kv_len_arr == [18] and len(h1.kv_indices) == 2
    assert attn.host_plan().kv_len_arr == [22]
    attn.select_plan_slot(1)
    assert attn.host_plan().kv_len_arr == [18]
