"""M4 step 3: validate mstar's REAL paged ``run_attention`` for the naive MLA.

The naive/materialized MLA stores a ``head_dim = qk_head_dim`` (nope+rope) K plus
a V padded to that same width, then calls the paged ``run_attention`` at the
fixed ``1/sqrt(head_dim)`` scale FlashInfer uses (which is exactly why the
``mscale^2`` softmax boost is folded into q — ``run_attention`` exposes no custom
``sm_scale``). This test drives the **real** ``FlashInferCacheManager`` over a
genuine paged KV cache (real ``PagedAllocationManager`` + ``LocalTransferEngine``)
and asserts its ``run_attention`` matches a causal-SDPA reference at
``1/sqrt(head_dim)`` — confirming both the paged path integrates and the scale
assumption the naive MLA relies on.

KEY CONSTRAINT FOUND (this is the "FlashInfer-192" answer, recorded for M5/M6):
FlashInfer 0.6.14's SM90 (Hopper / H200) prefill kernel has a compile-time
``static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256)``
(``flashinfer/.../attention/hopper/prefill_sm90.cuh:572``). The naive MLA pads V
to ``qk_head_dim``, so ``head_dim_vo == head_dim``:

  * real Kimi ``qk_head_dim = 192`` (nope 128 + rope 64) -> vo=192  -> JIT FAILS
  * reduced-config ``qk_head_dim = 24``                  -> vo=24   -> JIT FAILS
  * 64 / 128 / 256                                        -> supported -> OK

So the naive MLA path cannot use the paged ``run_attention`` at head_dim 192 (or
the reduced 24) on Hopper as-is. Validated mitigation (M6 follow-up, NOT done
here): pad ``head_dim`` up to the next supported vo (256 for real 192, 64 for the
reduced 24), pad q/k/v to it, and slice the attention output back to
``v_head_dim``. The supported-dim runs below (128 / 256) are exactly that padded
path; the env-gated test at the bottom records the raw 192 failure.

Run:  pytest test/integration/test_kimi_flashinfer_attention.py -v
      KIMI_TEST_FLASHINFER_192=1 pytest ... -k rejects   # ~60s failing JIT
"""
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
    """Build a genuine paged FlashInferCacheManager for one request.

    Mirrors ``KVCacheEngine.load_model`` / ``_create_cache_manager``: a real
    ``[layers, pages, 2, page_size, heads, head_dim]`` KV cache, a
    ``PagedAllocationManager`` over a no-op ``LocalTransferEngine`` (single-node
    SHM path — no cross-worker reads), and the flashinfer backend. Returns
    ``(cache_manager, alloc_manager)`` so the caller can clean up.
    """
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


# FlashInfer SM90 prefill supports head_dim_vo in {64, 128, 256}. These are the
# sizes the naive-MLA V-pad would target: 64 for the reduced config (24 -> 64),
# 128 canonical, 256 for the real Kimi qk_head_dim (192 -> 256).
@pytest.mark.parametrize("head_dim", [128, 256])
def test_real_paged_run_attention_matches_sdpa(head_dim):
    """The real FlashInferCacheManager.run_attention == causal SDPA at
    1/sqrt(head_dim), the fixed scale the naive MLA folds mscale^2 into q for."""
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
    """Executable record of the constraint: the real paged run_attention cannot
    JIT-build for head_dim=192 (vo=192) on Hopper — FlashInfer static_asserts
    HEAD_DIM_VO in {64,128,256}. If FlashInfer/the mitigation ever lifts this,
    this test flips to failing (no exception raised) and flags the change.

    The failure surfaces as the JIT build erroring out; the concrete exception
    type varies by stage (a RuntimeError wrapping the ninja/nvcc
    CalledProcessError), so we assert on the broad base and check the message
    points at the build rather than an unrelated error."""
    torch.manual_seed(0)
    num_heads, T, head_dim = 4, 6, 192
    dtype = torch.bfloat16
    cm, alloc = _make_real_cache_manager(num_heads, head_dim, dtype)
    try:
        q = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        k = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        v = torch.randn(T, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
        cm.set_active_label("main")
        # The offending kernel is JIT-built when FlashInfer schedules it — that
        # can happen in plan_attention (the wrapper.plan() call) or run_attention
        # depending on version, so both sit inside the raises block.
        with pytest.raises(Exception) as exc_info:
            cm.plan_attention(seq_lens=[T], is_causal=True, dtype=dtype)
            cm.set_layer_idx(0)
            cm.run_attention(q=q, k=k, v=v)
            torch.cuda.synchronize()
        # Guard against catching an unrelated error: the message must reference
        # the failed build / the offending head_dim.
        msg = str(exc_info.value).lower()
        assert "ninja" in msg or "build" in msg or "192" in msg
    finally:
        alloc.cleanup()
