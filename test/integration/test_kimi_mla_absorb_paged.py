import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.cache_manager import (
    MlaAbsorbCacheManager,
    MlaSdpaPlan,
    WorkspaceBufferManager,
    _PlanState,
    create_cache_manager,
)
from mstar.engine.kv_store import (
    KVCacheConfig,
    PagedAllocationManager,
    TransferEngineInfo,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the paged compressed-latent MLA backend runs on GPU",
)

DEVICE = torch.device("cuda")


def _make_latent_cache_manager(
    latent_width, dtype, softmax_scale, page_size=4, max_num_pages=64
):
    # 4D mla_absorb cache; small page_size forces multi-page coverage.
    kv_cache = torch.zeros(
        2, max_num_pages, page_size, latent_width,
        dtype=dtype, device=DEVICE,
    ).contiguous()
    kv_cfg = KVCacheConfig(
        num_layers=2, num_kv_heads=1, head_dim=latent_width,
        max_seq_len=page_size * max_num_pages, max_num_pages=max_num_pages,
        page_size=page_size, num_qo_heads=1,
        attention_backend="mla_absorb", softmax_scale=softmax_scale,
    )
    transfer_info = TransferEngineInfo(
        my_entity_id="kimi_mla_absorb_test",
        my_session_id="kimi_session",
        transfer_engine=LocalTransferEngine("localhost"),
    )
    alloc = PagedAllocationManager(
        config=kv_cfg, kv_cache=kv_cache, transfer_engine_info=transfer_info,
    )
    alloc.add_request("r0", ["main"])
    buffers = WorkspaceBufferManager(64 * 1024 * 1024, device=DEVICE)
    cm = create_cache_manager(
        request_ids=["r0"],
        active_labels_per_request={"r0": "main"},
        kv_cache=kv_cache,
        alloc_manager=alloc,
        buffer_manager=buffers,
        kv_cache_config=kv_cfg,
        device=DEVICE,
    )
    assert isinstance(cm, MlaAbsorbCacheManager)
    return cm, alloc


def _ref_mla_step(q_nope_new, q_pe_new, kv_c_all, k_pe_all, scale):
    sl, _H, L = q_nope_new.shape
    total = kv_c_all.shape[0]
    old_len = total - sl

    query = torch.cat([q_nope_new, q_pe_new], dim=-1)          # [sl,H,L+Drope]
    kv_c_h = kv_c_all.squeeze(1)                               # [total,L]
    k_pe_h = k_pe_all.squeeze(1)                               # [total,Drope]
    key = torch.cat([kv_c_h, k_pe_h], dim=-1)                 # [total,L+Drope]
    value = kv_c_h                                             # [total,L]

    qt = query.transpose(0, 1).float()                        # [H,sl,L+Drope]
    scores = torch.einsum("hqd,kd->hqk", qt, key.float()) * scale  # [H,sl,total]
    q_pos = old_len + torch.arange(sl, device=DEVICE)
    k_pos = torch.arange(total, device=DEVICE)
    mask = torch.where(
        k_pos[None, :] <= q_pos[:, None],
        0.0,
        torch.tensor(float("-inf"), device=DEVICE),
    )
    attn = (scores + mask).softmax(-1)
    out = torch.einsum("hqk,kd->hqd", attn, value.float())    # [H,sl,L]
    return out.transpose(0, 1).to(q_nope_new.dtype)           # [sl,H,L]


def _rand_step(sl, H, L, Drope, dtype):
    return (
        torch.randn(sl, H, L, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, H, Drope, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, L, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, Drope, device=DEVICE, dtype=dtype) * 0.1,
    )


def _run_prefill_then_decode(L, Drope, H, T, page_size):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    # Arbitrary MLA-style scale (the backend must apply exactly this value).
    scale = (L + Drope) ** -0.5 * 1.3

    cm, alloc = _make_latent_cache_manager(L + Drope, dtype, scale, page_size=page_size)
    try:
        cm.set_active_label("main")
        cm.set_layer_idx(0)

        assert T > page_size, "T must span >1 page to exercise page boundaries"
        q_nope, q_pe, kv_c, k_pe = _rand_step(T, H, L, Drope, dtype)
        cm.plan_attention(seq_lens=[T], is_causal=True, dtype=dtype)
        with torch.no_grad():
            got_prefill = cm.run_attention_mla(q_nope, q_pe, kv_c, k_pe)
        torch.cuda.synchronize()

        ref_prefill = _ref_mla_step(q_nope, q_pe, kv_c, k_pe, scale)
        assert got_prefill.shape == (T, H, L)
        torch.testing.assert_close(got_prefill, ref_prefill, rtol=2e-2, atol=2e-2)

        cm.advance_seq_lens()

        q_nope1, q_pe1, kv_c1, k_pe1 = _rand_step(1, H, L, Drope, dtype)
        cm.plan_attention(seq_lens=[1], is_causal=True, dtype=dtype)
        with torch.no_grad():
            got_decode = cm.run_attention_mla(q_nope1, q_pe1, kv_c1, k_pe1)
        torch.cuda.synchronize()

        kv_c_all = torch.cat([kv_c, kv_c1], dim=0)     # [T+1,1,L]
        k_pe_all = torch.cat([k_pe, k_pe1], dim=0)     # [T+1,1,Drope]
        ref_decode = _ref_mla_step(q_nope1, q_pe1, kv_c_all, k_pe_all, scale)
        assert got_decode.shape == (1, H, L)
        torch.testing.assert_close(got_decode, ref_decode, rtol=2e-2, atol=2e-2)
    finally:
        alloc.cleanup()


def test_paged_latent_mla_real_dims():
    _run_prefill_then_decode(L=512, Drope=64, H=2, T=6, page_size=4)


def test_paged_latent_mla_reduced_dims():
    _run_prefill_then_decode(L=32, Drope=8, H=4, T=6, page_size=4)


def test_sdpa_plan_clears_any_injected_wrapper():
    """``run_attention_mla`` picks its path with ``ps.wrapper is not None``, so the
    SDPA branch must clear the slot — otherwise a wrapper injected by CUDA-graph
    capture routes latent attention into a paged kernel that cannot read the 4-D
    cache. Mirrors how the paged path clears ``dense_gen``."""
    L, Drope = 32, 8  # reduced dims: no MLA kernel, so plan_attention takes SDPA
    cm, alloc = _make_latent_cache_manager(L + Drope, torch.bfloat16, 0.1)
    try:
        cm.set_active_label("main")
        # Stand in for a persistent wrapper injected via cuda_graph_plan_states.
        cm._plan_states["main"] = _PlanState(wrapper=object())

        cm.plan_attention(seq_lens=[4], is_causal=True, dtype=torch.bfloat16)

        ps = cm._plan_states["main"]
        assert ps.wrapper is None, "SDPA plan left a stale wrapper in the plan state"
        assert isinstance(ps.mla, MlaSdpaPlan)
        assert [r.seq_len for r in ps.mla.requests] == [4]
    finally:
        alloc.cleanup()
