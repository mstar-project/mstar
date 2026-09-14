"""Real FlashInfer paged attention through the attention resource.

Pins two things about the paged backend the naive-MLA path rides on: that its
output matches a causal SDPA reference at the head dims Kimi uses, and that
head_dim=192 is rejected rather than silently mis-serving (the reason naive MLA
pads to ``padded_head_dim``).
"""

import os

import pytest
import torch
from kimi_harness import (
    DEVICE,
    build_resources,
    cleanup,
    ingest,
    paged_specs,
    step,
)
from kimi_reference import sdpa_causal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real FlashInfer paged attention needs a GPU",
)


def _resources(num_heads, head_dim):
    return build_resources(
        paged_specs(num_kv_heads=num_heads, head_dim=head_dim),
        entity_id="kimi_flashinfer_test",
    )


@pytest.mark.parametrize("head_dim", [128, 256])
def test_real_paged_run_attention_matches_sdpa(head_dim):
    torch.manual_seed(0)
    num_heads, seq_len = 4, 6
    dtype = torch.bfloat16
    resources = _resources(num_heads, head_dim)
    kv, attn = resources["kv_cache"], resources["attn"]
    try:
        ingest(resources, "r0")
        q, k, v = (
            torch.randn(seq_len, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
            for _ in range(3)
        )
        with step(resources, {"r0": seq_len}):
            assert attn.requires_kv_write
            kv.write_kv(k, v, layer_idx=0)
            got = attn.run(q, "main", kv.layer_view(0))
        torch.cuda.synchronize()

        expected = sdpa_causal(q, k, v, head_dim ** -0.5)
        assert got.shape == (seq_len, num_heads, head_dim)
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    finally:
        cleanup(resources)


@pytest.mark.skipif(
    os.environ.get("KIMI_TEST_FLASHINFER_192") != "1",
    reason="opt-in (~60s failing JIT): set KIMI_TEST_FLASHINFER_192=1 to record "
           "the head_dim=192 SM90 static_assert rejection",
)
def test_flashinfer_rejects_head_dim_192():
    torch.manual_seed(0)
    num_heads, seq_len, head_dim = 4, 6, 192
    dtype = torch.bfloat16
    resources = _resources(num_heads, head_dim)
    kv, attn = resources["kv_cache"], resources["attn"]
    try:
        ingest(resources, "r0")
        q, k, v = (
            torch.randn(seq_len, num_heads, head_dim, device=DEVICE, dtype=dtype) * 0.1
            for _ in range(3)
        )
        # FlashInfer may JIT the failing kernel in either plan or run.
        with pytest.raises(Exception) as exc_info:  # noqa: B017, PT011
            with step(resources, {"r0": seq_len}):
                kv.write_kv(k, v, layer_idx=0)
                attn.run(q, "main", kv.layer_view(0))
                torch.cuda.synchronize()
        msg = str(exc_info.value).lower()
        assert "ninja" in msg or "build" in msg or "192" in msg
    finally:
        cleanup(resources)
