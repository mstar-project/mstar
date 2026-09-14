"""The FlashInfer MLA kernel fast path, on the hardware that has it.

sm90-only. Three things: the kernel agrees with the eager SDPA fallback and the
reference (so the two paths are interchangeable), batched decode addresses each
request's own pages, and the wrapper replays correctly from a captured graph.
"""

import pytest
import torch
from kimi_harness import (
    DEVICE,
    build_resources,
    cleanup,
    ingest,
    latent_specs,
    step,
)
from kimi_reference import ref_mla_latent_step

from mstar.engine.resources.attn.mla import _mla_kernel_available

_IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9

pytestmark = pytest.mark.skipif(
    not _IS_SM90,
    reason="the FlashInfer MLA kernel fast path requires a Hopper (sm90) GPU",
)

CKV, KPE = 512, 64
LATENT = CKV + KPE


def _resources(scale, mla_ckv_dim, num_heads, page_size=4):
    return build_resources(
        latent_specs(
            latent_width=LATENT,
            softmax_scale=scale,
            mla_ckv_dim=mla_ckv_dim,
            num_qo_heads=num_heads,
            page_size=page_size,
        ),
        entity_id="kimi_mla_kernel_test",
    )


def _rand_step(sl, num_heads, dtype=torch.bfloat16):
    return (
        torch.randn(sl, num_heads, CKV, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, num_heads, KPE, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, CKV, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, KPE, device=DEVICE, dtype=dtype) * 0.1,
    )


def _attend(resources, latents, spans):
    kv, attn = resources["kv_cache"], resources["attn"]
    q_nope, q_pe, kv_c, k_pe = latents
    with step(resources, spans):
        kv.set_layer_idx(0)
        kv.write_latent(torch.cat([kv_c, k_pe], dim=-1).squeeze(1))
        with torch.no_grad():
            out = attn.run_mla(q_nope, q_pe, kv.layer_view(0), label="main")
    torch.cuda.synchronize()
    return out


def _prefill_then_decode(resources, seq_len, num_heads, scale):
    ingest(resources, "r0")
    prefill = _rand_step(seq_len, num_heads)
    got_prefill = _attend(resources, prefill, {"r0": seq_len})
    ref_prefill = ref_mla_latent_step(*prefill, scale)

    decode = _rand_step(1, num_heads)
    got_decode = _attend(resources, decode, {"r0": 1})
    ref_decode = ref_mla_latent_step(
        decode[0], decode[1],
        torch.cat([prefill[2], decode[2]], dim=0),
        torch.cat([prefill[3], decode[3]], dim=0),
        scale,
    )
    return (got_prefill, ref_prefill), (got_decode, ref_decode)


def test_kernel_matches_sdpa_and_reference():
    """The kernel and the fallback are interchangeable, so run the same seeded
    step through each and check both against the reference and each other."""
    num_heads, seq_len = 2, 6
    scale = LATENT ** -0.5 * 1.3
    assert _mla_kernel_available(CKV, KPE, 9) is True

    torch.manual_seed(0)
    kernel_res = _resources(scale, CKV, num_heads)
    try:
        assert kernel_res["attn"].mla_kernel_available() is True
        (kp, refp), (kd, refd) = _prefill_then_decode(
            kernel_res, seq_len, num_heads, scale
        )
        assert kernel_res["attn"]._current_plan_states["main"].wrapper is not None
    finally:
        cleanup(kernel_res)

    torch.manual_seed(0)
    # mla_ckv_dim=None makes the predicate decline, forcing SDPA on this same box
    sdpa_res = _resources(scale, None, num_heads)
    try:
        assert sdpa_res["attn"].mla_kernel_available() is False
        (sp, _), (sd, _) = _prefill_then_decode(
            sdpa_res, seq_len, num_heads, scale
        )
        assert sdpa_res["attn"]._current_plan_states["main"].wrapper is None
    finally:
        cleanup(sdpa_res)

    assert kp.shape == (seq_len, num_heads, CKV) and kd.shape == (1, num_heads, CKV)
    for got in (kp, sp):
        torch.testing.assert_close(got, refp, rtol=2e-2, atol=2e-2)
    for got in (kd, sd):
        torch.testing.assert_close(got, refd, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kp, sp, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kd, sd, rtol=2e-2, atol=2e-2)


def test_kernel_batched_decode():
    """Two requests at different lengths: each decode row must attend its own
    pages, which is what the per-request kv_len the plan builds is for."""
    num_heads = 2
    scale = LATENT ** -0.5
    lens = {"r0": 5, "r1": 9}
    rids = list(lens)

    torch.manual_seed(1)
    resources = _resources(scale, CKV, num_heads)
    attn = resources["attn"]
    try:
        ingest(resources, *rids)
        prefill = {rid: _rand_step(sl, num_heads) for rid, sl in lens.items()}
        packed = tuple(
            torch.cat([prefill[rid][i] for rid in rids], dim=0) for i in range(4)
        )
        _attend(resources, packed, lens)
        assert attn._current_plan_states["main"].wrapper is not None

        decode = {rid: _rand_step(1, num_heads) for rid in rids}
        packed_decode = tuple(
            torch.cat([decode[rid][i] for rid in rids], dim=0) for i in range(4)
        )
        got = _attend(resources, packed_decode, dict.fromkeys(rids, 1))
        assert got.shape == (len(rids), num_heads, CKV)

        for i, rid in enumerate(rids):
            ref = ref_mla_latent_step(
                decode[rid][0], decode[rid][1],
                torch.cat([prefill[rid][2], decode[rid][2]], dim=0),
                torch.cat([prefill[rid][3], decode[rid][3]], dim=0),
                scale,
            )
            torch.testing.assert_close(got[i:i + 1], ref, rtol=2e-2, atol=2e-2)
    finally:
        cleanup(resources)


def test_mla_wrapper_cuda_graph_capture_replay():
    """The captured region is the latent scatter plus the kernel, both reading
    static buffers — the scatter indices are the KV plan's, so this builds them
    the way ``KVPlanState`` does and replays two decode steps through one graph.
    """
    from mstar.engine.resources.attn.wrappers import FlashInferMLAWrapper

    bs, num_heads, page_size, max_pages = 2, 2, 4, 64
    scale = LATENT ** -0.5 * 1.1
    dtype = torch.bfloat16

    cache = torch.zeros(max_pages, page_size, LATENT, device=DEVICE, dtype=dtype)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
    wrapper = FlashInferMLAWrapper(
        workspace, num_heads=num_heads, head_dim_ckv=CKV, head_dim_kpe=KPE,
        page_size=page_size, sm_scale=scale, batch_size=bs,
        max_num_pages=max_pages, device=DEVICE, use_cuda_graph=True,
    )

    # decode stays within fixed pages; only kv_len and the scatter offsets move
    req_pages = [[0, 1], [2, 3, 4]]
    prefill = [6, 9]
    torch.manual_seed(7)
    prefix = []
    for r in range(bs):
        lat = torch.randn(prefill[r], LATENT, device=DEVICE, dtype=dtype) * 0.1
        for t in range(prefill[r]):
            cache[req_pages[r][t // page_size], t % page_size] = lat[t]
        prefix.append(lat)

    kv_indptr = torch.tensor(
        [0, len(req_pages[0]), len(req_pages[0]) + len(req_pages[1])],
        device=DEVICE, dtype=torch.int32,
    )
    kv_indices = torch.tensor(
        req_pages[0] + req_pages[1], device=DEVICE, dtype=torch.int32
    )
    qo_indptr = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int32)

    q_nope_s = torch.zeros(bs, num_heads, CKV, device=DEVICE, dtype=dtype)
    q_pe_s = torch.zeros(bs, num_heads, KPE, device=DEVICE, dtype=dtype)
    latent_s = torch.zeros(bs, LATENT, device=DEVICE, dtype=dtype)
    # the scatter's static index buffers, as the KV resource holds them
    page_s = torch.zeros(bs, dtype=torch.long, device=DEVICE)
    offset_s = torch.zeros(bs, dtype=torch.long, device=DEVICE)

    def plan_step(pos):
        """Plan for a decode landing at ``prefill[r] + pos - 1`` per request."""
        kv_len_arr = torch.tensor(
            [prefill[r] + pos for r in range(bs)], device=DEVICE, dtype=torch.int32,
        )
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_arr,
            causal=True, dtype=dtype,
        )
        slot = [prefill[r] + pos - 1 for r in range(bs)]
        page_s.copy_(torch.tensor(
            [req_pages[r][slot[r] // page_size] for r in range(bs)], device=DEVICE,
        ))
        offset_s.copy_(torch.tensor(
            [slot[r] % page_size for r in range(bs)], device=DEVICE,
        ))

    def fill_inputs(seed):
        torch.manual_seed(seed)
        q_nope_s.copy_(torch.randn(bs, num_heads, CKV, device=DEVICE, dtype=dtype) * 0.1)
        q_pe_s.copy_(torch.randn(bs, num_heads, KPE, device=DEVICE, dtype=dtype) * 0.1)
        latent_s.copy_(torch.randn(bs, LATENT, device=DEVICE, dtype=dtype) * 0.1)

    def region():
        cache[page_s, offset_s] = latent_s
        return wrapper.run(q_nope_s, q_pe_s, cache[..., :CKV], cache[..., CKV:])

    plan_step(1)
    fill_inputs(100)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            region()
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = region()

    decode_hist = [[] for _ in range(bs)]
    prev = None
    for pos in (1, 2):
        fill_inputs(pos)
        plan_step(pos)
        graph.replay()
        torch.cuda.synchronize()
        for r in range(bs):
            decode_hist[r].append(latent_s[r:r + 1].clone())
        got = out_static.clone()

        ref = torch.empty(bs, num_heads, CKV, device=DEVICE, dtype=dtype)
        for r in range(bs):
            all_lat = torch.cat([prefix[r]] + decode_hist[r], dim=0)
            ref[r:r + 1] = ref_mla_latent_step(
                q_nope_s[r:r + 1], q_pe_s[r:r + 1],
                all_lat[:, :CKV].unsqueeze(1), all_lat[:, CKV:].unsqueeze(1), scale,
            )
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        if prev is not None:
            # distinct inputs prove replay reads the static buffers
            assert not torch.allclose(got, prev, atol=1e-3)
        prev = got


def test_probe_declines_reduced_dims():
    assert _mla_kernel_available(CKV, KPE, 9) is True    # real dims, Hopper
    assert _mla_kernel_available(32, 8, 9) is False      # reduced dims
    assert _mla_kernel_available(CKV, KPE, 8) is False   # pre-sm90
    assert _mla_kernel_available(CKV, KPE, 10) is False  # Blackwell (trtllm path)
