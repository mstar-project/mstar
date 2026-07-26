"""Phase-B GPU check: the REAL paged compressed-latent MLA backend.

Drives ``MlaAbsorbCacheManager.run_attention_mla`` over a genuine 4D latent
paged cache (real ``PagedAllocationManager`` + ``create_cache_manager``), with
SYNTHETIC random ``q_nope/q_pe/kv_c/k_pe`` — no ``KimiMLAAttention`` needed. The
manager writes ``cat([kv_c, k_pe])`` as one latent vector per token into the
paged cache at its (page, offset), then per request gathers its full cached
latent and runs a causal SDPA (query = ``cat([q_nope, q_pe])``, key = the full
latent, value = its first ``L`` dims) at ``kv_cache_config.softmax_scale``.

The reference is INDEPENDENT of the paging machinery: it accumulates every
``(kv_c, k_pe)`` seen so far into contiguous tensors and runs the same causal
SDPA (no pages, no scatter/gather). Matching it across a multi-page prefill and a
following decode step proves the paged scatter/gather + causal mask + scale are
correct — at real Kimi latent dims (L=512, Drope=64) and reduced dims.

Run:  pytest test/integration/test_kimi_mla_absorb_paged.py -v
"""
import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.cache_manager import (
    MlaAbsorbCacheManager,
    WorkspaceBufferManager,
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


# --------------------------------------------------------------------------
# Real paged latent cache manager (mirrors test_kimi_mla_paged.py, but 4D).
# --------------------------------------------------------------------------

def _make_latent_cache_manager(
    latent_width, dtype, softmax_scale, page_size=4, max_num_pages=64
):
    # 4D latent cache: [num_layers, max_pages, page_size, latent_width]
    # (the shape KVCacheEngine.load_model allocates for attention_backend
    # "mla_absorb"). Small page_size so a handful of tokens spans >1 page.
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


# --------------------------------------------------------------------------
# Independent reference: accumulate all (kv_c, k_pe), causal SDPA (no paging).
# --------------------------------------------------------------------------

def _ref_mla_step(q_nope_new, q_pe_new, kv_c_all, k_pe_all, scale):
    """The intended MLA output for the NEW query tokens.

    q_nope_new [sl,H,L], q_pe_new [sl,H,Drope] are just this step's queries;
    kv_c_all [total,1,L], k_pe_all [total,1,Drope] are EVERY latent cached so far
    (including this step's). Query j sits at absolute position old_len+j and
    attends to cached 0..old_len+j; value is the kv_c (first L) part.
    """
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
    """Drive a multi-page prefill + a decode step through the real paged backend
    and compare each to the independent accumulate-everything reference."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    # Arbitrary MLA-style scale (the backend must apply exactly this value).
    scale = (L + Drope) ** -0.5 * 1.3

    cm, alloc = _make_latent_cache_manager(L + Drope, dtype, scale, page_size=page_size)
    try:
        cm.set_active_label("main")
        cm.set_layer_idx(0)

        # ---- prefill: T tokens spanning more than one page ----
        assert T > page_size, "T must span >1 page to exercise page boundaries"
        q_nope, q_pe, kv_c, k_pe = _rand_step(T, H, L, Drope, dtype)
        cm.plan_attention(seq_lens=[T], is_causal=True, dtype=dtype)
        with torch.no_grad():
            got_prefill = cm.run_attention_mla(q_nope, q_pe, kv_c, k_pe)
        torch.cuda.synchronize()

        ref_prefill = _ref_mla_step(q_nope, q_pe, kv_c, k_pe, scale)
        assert got_prefill.shape == (T, H, L)
        torch.testing.assert_close(got_prefill, ref_prefill, rtol=2e-2, atol=2e-2)

        # Advance seq_len so the decode step sees the T cached tokens.
        cm.advance_seq_lens()

        # ---- decode: 1 new token attends over all T+1 cached ----
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
    """Real Kimi MLA latent dims: L=512, Drope=64, H=2. Prefill (6 tokens over a
    page_size-4 cache = 2 pages) + a decode step."""
    _run_prefill_then_decode(L=512, Drope=64, H=2, T=6, page_size=4)


def test_paged_latent_mla_reduced_dims():
    """Reduced dims: L=32, Drope=8, H=4. Same multi-page prefill + decode step."""
    _run_prefill_then_decode(L=32, Drope=8, H=4, T=6, page_size=4)
