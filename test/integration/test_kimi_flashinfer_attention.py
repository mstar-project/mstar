import os

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.cache_manager import WorkspaceBufferManager, create_cache_manager
from mstar.engine.kv_store import (
    KVCacheConfig,
    PagedAllocationManager,
    TransferEngineInfo,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real FlashInfer paged attention needs a GPU",
)

DEVICE = torch.device("cuda")


def _make_real_cache_manager(num_heads, head_dim, dtype, page_size=128, max_num_pages=8):
    kv_cache = torch.zeros(
        2, max_num_pages, 2, page_size, num_heads, head_dim,
        dtype=dtype, device=DEVICE,
    ).contiguous()
    kv_cfg = KVCacheConfig(
        num_layers=2, num_kv_heads=num_heads, head_dim=head_dim,
        max_seq_len=page_size * max_num_pages, max_num_pages=max_num_pages,
        page_size=page_size, num_qo_heads=num_heads,
    )
    transfer_info = TransferEngineInfo(
        my_entity_id="kimi_flashinfer_test",
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
    return cm, alloc


def _sdpa_causal(q, k, v, scale):
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))  # (H,T,D)
    T = q.shape[0]
    causal = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


@pytest.mark.parametrize("head_dim", [128, 256])
def test_real_paged_run_attention_matches_sdpa(head_dim):
    torch.manual_seed(0)
    num_heads, T = 4, 6
    dtype = torch.bfloat16
    cm, alloc = _make_real_cache_manager(num_heads, head_dim, dtype)
    try:
        q = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        k = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        v = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1

        cm.set_active_label("main")
        cm.plan_attention(seq_lens=[T], is_causal=True, dtype=dtype)
        cm.set_layer_idx(0)
        got = cm.run_attention(q=q, k=k, v=v)
        torch.cuda.synchronize()

        expected = _sdpa_causal(q, k, v, head_dim ** -0.5)
        assert got.shape == (T, num_heads, head_dim)
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    finally:
        alloc.cleanup()


@pytest.mark.skipif(
    os.environ.get("KIMI_TEST_FLASHINFER_192") != "1",
    reason="opt-in (~60s failing JIT): set KIMI_TEST_FLASHINFER_192=1 to record "
           "the head_dim=192 SM90 static_assert rejection",
)
def test_flashinfer_rejects_head_dim_192():
    torch.manual_seed(0)
    num_heads, T, head_dim = 4, 6, 192
    dtype = torch.bfloat16
    cm, alloc = _make_real_cache_manager(num_heads, head_dim, dtype)
    try:
        q = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        k = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        v = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        cm.set_active_label("main")
        # FlashInfer may JIT the failing kernel in plan_attention or run_attention.
        with pytest.raises(Exception) as exc_info:
            cm.plan_attention(seq_lens=[T], is_causal=True, dtype=dtype)
            cm.set_layer_idx(0)
            cm.run_attention(q=q, k=k, v=v)
            torch.cuda.synchronize()
        # Guard against catching an unrelated error.
        msg = str(exc_info.value).lower()
        assert "ninja" in msg or "build" in msg or "192" in msg
    finally:
        alloc.cleanup()
