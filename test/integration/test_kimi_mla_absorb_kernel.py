import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.cache_manager import (
    MlaAbsorbCacheManager,
    _mla_kernel_available,
    create_cache_manager,
)
from mstar.engine.kv_store import (
    KVCacheConfig,
    PagedAllocationManager,
    TransferEngineInfo,
)

_IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9

pytestmark = pytest.mark.skipif(
    not _IS_SM90,
    reason="the FlashInfer MLA kernel fast path requires a Hopper (sm90) GPU",
)

DEVICE = torch.device("cuda")
L, DROPE = 512, 64


def _make_cm(softmax_scale, ckv_dim, request_ids, page_size=4, max_num_pages=64):
    latent_width = L + DROPE
    kv_cache = torch.zeros(
        2, max_num_pages, page_size, latent_width, dtype=torch.bfloat16, device=DEVICE,
    ).contiguous()
    kv_cfg = KVCacheConfig(
        num_layers=2, num_kv_heads=1, head_dim=latent_width,
        max_seq_len=page_size * max_num_pages, max_num_pages=max_num_pages,
        page_size=page_size, num_qo_heads=1,
        attention_backend="mla_absorb", softmax_scale=softmax_scale,
        mla_ckv_dim=ckv_dim,
    )
    transfer_info = TransferEngineInfo(
        my_entity_id="kimi_mla_kernel_test",
        my_session_id="kimi_session",
        transfer_engine=LocalTransferEngine("localhost"),
    )
    alloc = PagedAllocationManager(
        config=kv_cfg, kv_cache=kv_cache, transfer_engine_info=transfer_info,
    )
    for rid in request_ids:
        alloc.add_request(rid, ["main"])
    from mstar.engine.cache_manager import WorkspaceBufferManager
    buffers = WorkspaceBufferManager(128 * 1024 * 1024, device=DEVICE)
    cm = create_cache_manager(
        request_ids=list(request_ids),
        active_labels_per_request={rid: "main" for rid in request_ids},
        kv_cache=kv_cache,
        alloc_manager=alloc,
        buffer_manager=buffers,
        kv_cache_config=kv_cfg,
        device=DEVICE,
    )
    assert isinstance(cm, MlaAbsorbCacheManager)
    cm.set_active_label("main")
    cm.set_layer_idx(0)
    return cm, alloc

def _ref_mla_step(q_nope_new, q_pe_new, kv_c_all, k_pe_all, scale):
    sl, _H, _L = q_nope_new.shape
    total = kv_c_all.shape[0]
    old_len = total - sl
    query = torch.cat([q_nope_new, q_pe_new], dim=-1)
    key = torch.cat([kv_c_all.squeeze(1), k_pe_all.squeeze(1)], dim=-1)  # [total,L+Drope]
    value = kv_c_all.squeeze(1)
    qt = query.transpose(0, 1).float()
    scores = torch.einsum("hqd,kd->hqk", qt, key.float()) * scale
    q_pos = old_len + torch.arange(sl, device=DEVICE)
    k_pos = torch.arange(total, device=DEVICE)
    mask = torch.where(k_pos[None, :] <= q_pos[:, None], 0.0,
                       torch.tensor(float("-inf"), device=DEVICE))
    attn = (scores + mask).softmax(-1)
    out = torch.einsum("hqk,kd->hqd", attn, value.float())
    return out.transpose(0, 1).to(q_nope_new.dtype)


def _rand_step(sl, H, dtype=torch.bfloat16):
    return (
        torch.randn(sl, H, L, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, H, DROPE, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, L, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, DROPE, device=DEVICE, dtype=dtype) * 0.1,
    )


def _run_single_req(cm, alloc, T, H, scale):
    q_nope, q_pe, kv_c, k_pe = _rand_step(T, H)
    cm.plan_attention(seq_lens=[T], is_causal=True, dtype=torch.bfloat16)
    with torch.no_grad():
        got_prefill = cm.run_attention_mla(q_nope, q_pe, kv_c, k_pe)
    torch.cuda.synchronize()
    ref_prefill = _ref_mla_step(q_nope, q_pe, kv_c, k_pe, scale)
    cm.advance_seq_lens()

    q_nope1, q_pe1, kv_c1, k_pe1 = _rand_step(1, H)
    cm.plan_attention(seq_lens=[1], is_causal=True, dtype=torch.bfloat16)
    with torch.no_grad():
        got_decode = cm.run_attention_mla(q_nope1, q_pe1, kv_c1, k_pe1)
    torch.cuda.synchronize()
    ref_decode = _ref_mla_step(
        q_nope1, q_pe1,
        torch.cat([kv_c, kv_c1], 0), torch.cat([k_pe, k_pe1], 0), scale,
    )
    return (got_prefill, ref_prefill), (got_decode, ref_decode)


def test_kernel_matches_sdpa_and_reference():
    H, T, ps = 2, 6, 4
    scale = (L + DROPE) ** -0.5 * 1.3
    assert _mla_kernel_available(L, DROPE, 9) is True

    torch.manual_seed(0)
    cm_k, alloc_k = _make_cm(scale, ckv_dim=L, request_ids=["r0"], page_size=ps)
    try:
        (kp, refp), (kd, refd) = _run_single_req(cm_k, alloc_k, T, H, scale)
    finally:
        alloc_k.cleanup()

    torch.manual_seed(0)
    cm_s, alloc_s = _make_cm(scale, ckv_dim=None, request_ids=["r0"], page_size=ps)
    try:
        (sp, _refp), (sd, _refd) = _run_single_req(cm_s, alloc_s, T, H, scale)
    finally:
        alloc_s.cleanup()

    assert cm_k._plan_states["main"].wrapper is not None
    assert cm_s._plan_states["main"].wrapper is None

    assert kp.shape == (T, H, L) and kd.shape == (1, H, L)
    torch.testing.assert_close(kp, refp, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kd, refd, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sp, refp, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sd, refd, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kp, sp, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kd, sd, rtol=2e-2, atol=2e-2)


def test_kernel_batched_decode():
    H, ps = 2, 4
    scale = (L + DROPE) ** -0.5
    lens = [5, 9]
    rids = ["r0", "r1"]

    torch.manual_seed(1)
    cm, alloc = _make_cm(scale, ckv_dim=L, request_ids=rids, page_size=ps)
    try:
        pref = [_rand_step(sl, H) for sl in lens]
        q_nope = torch.cat([p[0] for p in pref], 0)
        q_pe = torch.cat([p[1] for p in pref], 0)
        kv_c = torch.cat([p[2] for p in pref], 0)
        k_pe = torch.cat([p[3] for p in pref], 0)
        cm.plan_attention(seq_lens=lens, is_causal=True, dtype=torch.bfloat16)
        with torch.no_grad():
            cm.run_attention_mla(q_nope, q_pe, kv_c, k_pe)
        torch.cuda.synchronize()
        cm.advance_seq_lens()
        assert cm._plan_states["main"].wrapper is not None

        dec = [_rand_step(1, H) for _ in lens]
        dq_nope = torch.cat([d[0] for d in dec], 0)
        dq_pe = torch.cat([d[1] for d in dec], 0)
        dkv_c = torch.cat([d[2] for d in dec], 0)
        dk_pe = torch.cat([d[3] for d in dec], 0)
        cm.plan_attention(seq_lens=[1, 1], is_causal=True, dtype=torch.bfloat16)
        with torch.no_grad():
            got = cm.run_attention_mla(dq_nope, dq_pe, dkv_c, dk_pe)
        torch.cuda.synchronize()
        assert got.shape == (2, H, L)

        for i in range(2):
            kv_c_all = torch.cat([pref[i][2], dec[i][2]], 0)
            k_pe_all = torch.cat([pref[i][3], dec[i][3]], 0)
            ref = _ref_mla_step(dec[i][0], dec[i][1], kv_c_all, k_pe_all, scale)
            torch.testing.assert_close(got[i:i + 1], ref, rtol=2e-2, atol=2e-2)
    finally:
        alloc.cleanup()


def test_mla_wrapper_cuda_graph_capture_replay():
    from mstar.utils.flashinfer_utils import FlashInferMLAWrapper

    bs, H, ps, max_pages = 2, 2, 4, 64
    latent_w = L + DROPE
    scale = latent_w ** -0.5 * 1.1
    dtype = torch.bfloat16

    cache = torch.zeros(max_pages, ps, latent_w, device=DEVICE, dtype=dtype)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
    wrapper = FlashInferMLAWrapper(
        ws, num_heads=H, head_dim_ckv=L, head_dim_kpe=DROPE, page_size=ps,
        sm_scale=scale, batch_size=bs, max_num_pages=max_pages,
        max_total_tokens=bs, device=DEVICE, use_cuda_graph=True,
    )

    # Decode steps stay within fixed pages; only kv_len and scatter offsets advance.
    req_pages = [[0, 1], [2, 3, 4]]
    prefill = [6, 9]
    torch.manual_seed(7)
    prefix = []  # [prefill_r, latent_w] latents already resident per request
    for r in range(bs):
        lat = torch.randn(prefill[r], latent_w, device=DEVICE, dtype=dtype) * 0.1
        for t in range(prefill[r]):
            cache[req_pages[r][t // ps], t % ps] = lat[t]
        prefix.append(lat)

    kv_indptr = torch.tensor(
        [0, len(req_pages[0]), len(req_pages[0]) + len(req_pages[1])],
        device=DEVICE, dtype=torch.int32,
    )
    kv_indices = torch.tensor(req_pages[0] + req_pages[1], device=DEVICE, dtype=torch.int32)
    qo_indptr = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int32)

    q_nope_s = torch.zeros(bs, H, L, device=DEVICE, dtype=dtype)
    q_pe_s = torch.zeros(bs, H, DROPE, device=DEVICE, dtype=dtype)
    latent_s = torch.zeros(bs, latent_w, device=DEVICE, dtype=dtype)

    def plan_step(step):  # step 1 -> position prefill_r, step 2 -> prefill_r+1
        kv_len_arr = torch.tensor(
            [prefill[r] + step for r in range(bs)], device=DEVICE, dtype=torch.int32,
        )
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, causal=True, dtype=dtype)

    def fill_inputs(seed):
        torch.manual_seed(seed)
        q_nope_s.copy_(torch.randn(bs, H, L, device=DEVICE, dtype=dtype) * 0.1)
        q_pe_s.copy_(torch.randn(bs, H, DROPE, device=DEVICE, dtype=dtype) * 0.1)
        latent_s.copy_(torch.randn(bs, latent_w, device=DEVICE, dtype=dtype) * 0.1)

    plan_step(1)
    fill_inputs(100)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            wrapper.set_latent(cache, latent_s)
            wrapper.run(q_nope_s, q_pe_s, cache[..., :L], cache[..., L:])
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wrapper.set_latent(cache, latent_s)
        out_static = wrapper.run(q_nope_s, q_pe_s, cache[..., :L], cache[..., L:])

    decode_hist = [[] for _ in range(bs)]  # decode latents scattered so far, per req
    prev = None
    for step in (1, 2):
        fill_inputs(step)
        plan_step(step)
        g.replay()
        torch.cuda.synchronize()
        for r in range(bs):
            decode_hist[r].append(latent_s[r:r + 1].clone())
        got = out_static.clone()

        ref = torch.empty(bs, H, L, device=DEVICE, dtype=dtype)
        for r in range(bs):
            all_lat = torch.cat([prefix[r]] + decode_hist[r], dim=0)  # [prefill+step, w]
            ref[r:r + 1] = _ref_mla_step(
                q_nope_s[r:r + 1], q_pe_s[r:r + 1],
                all_lat[:, :L].unsqueeze(1), all_lat[:, L:].unsqueeze(1), scale,
            )
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        if prev is not None:
            # Distinct inputs prove replay reads static buffers, not baked values.
            assert not torch.allclose(got, prev, atol=1e-3)
        prev = got


def test_probe_declines_reduced_dims():
    assert _mla_kernel_available(L, DROPE, 9) is True     # real dims, Hopper
    assert _mla_kernel_available(32, 8, 9) is False        # reduced dims
    assert _mla_kernel_available(L, DROPE, 8) is False     # pre-sm90
    assert _mla_kernel_available(L, DROPE, 10) is False    # Blackwell (wants trtllm path)
