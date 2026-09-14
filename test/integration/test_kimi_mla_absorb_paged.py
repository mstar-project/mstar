"""Absorbed MLA over the real MLA-layout cache, both paths.

Drives prefill then decode across a page boundary and checks each against the
compressed-latent reference. Parametrized over real Kimi dims (which take the
FlashInfer MLA kernel on sm90) and reduced dims (which take the eager SDPA
fallback), so one body covers both paths the backend can pick.
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

from mstar.engine.resources.attn.mla import MlaAbsorbManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the paged compressed-latent MLA backend runs on GPU",
)


def _resources(ckv, kpe, num_heads, scale, page_size, mla_ckv_dim=...):
    specs = latent_specs(
        latent_width=ckv + kpe,
        softmax_scale=scale,
        mla_ckv_dim=ckv if mla_ckv_dim is ... else mla_ckv_dim,
        num_qo_heads=num_heads,
        page_size=page_size,
    )
    resources = build_resources(specs, entity_id="kimi_mla_absorb_test")
    assert isinstance(resources["attn"], MlaAbsorbManager)
    return resources


def _rand_step(sl, num_heads, ckv, kpe, dtype):
    return (
        torch.randn(sl, num_heads, ckv, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, num_heads, kpe, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, ckv, device=DEVICE, dtype=dtype) * 0.1,
        torch.randn(sl, 1, kpe, device=DEVICE, dtype=dtype) * 0.1,
    )


def _attend(resources, latents, spans):
    """One step: write the latents through the KV resource, then attend."""
    kv, attn = resources["kv_cache"], resources["attn"]
    q_nope, q_pe, kv_c, k_pe = latents
    with step(resources, spans):
        kv.set_layer_idx(0)
        kv.write_latent(torch.cat([kv_c, k_pe], dim=-1).squeeze(1))
        with torch.no_grad():
            out = attn.run_mla(q_nope, q_pe, kv.layer_view(0), label="main")
    torch.cuda.synchronize()
    return out


def _prefill_then_decode(ckv, kpe, num_heads, seq_len, page_size):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    # Arbitrary MLA-style scale; the backend must apply exactly this value.
    scale = (ckv + kpe) ** -0.5 * 1.3
    resources = _resources(ckv, kpe, num_heads, scale, page_size)
    try:
        ingest(resources, "r0")
        assert seq_len > page_size, "must span >1 page to exercise page boundaries"

        prefill = _rand_step(seq_len, num_heads, ckv, kpe, dtype)
        got_prefill = _attend(resources, prefill, {"r0": seq_len})
        ref_prefill = ref_mla_latent_step(*prefill, scale)
        assert got_prefill.shape == (seq_len, num_heads, ckv)
        torch.testing.assert_close(got_prefill, ref_prefill, rtol=2e-2, atol=2e-2)

        decode = _rand_step(1, num_heads, ckv, kpe, dtype)
        got_decode = _attend(resources, decode, {"r0": 1})
        ref_decode = ref_mla_latent_step(
            decode[0], decode[1],
            torch.cat([prefill[2], decode[2]], dim=0),
            torch.cat([prefill[3], decode[3]], dim=0),
            scale,
        )
        assert got_decode.shape == (1, num_heads, ckv)
        torch.testing.assert_close(got_decode, ref_decode, rtol=2e-2, atol=2e-2)
        return resources["attn"]
    finally:
        cleanup(resources)


def test_paged_latent_mla_real_dims():
    """Real Kimi dims: the kernel serves these on sm90, SDPA everywhere else."""
    attn = _prefill_then_decode(
        ckv=512, kpe=64, num_heads=2, seq_len=6, page_size=4,
    )
    on_sm90 = torch.cuda.get_device_capability()[0] == 9
    assert attn.mla_kernel_available() is on_sm90


def test_paged_latent_mla_reduced_dims():
    """Reduced dims the kernel cannot serve: always the SDPA fallback."""
    attn = _prefill_then_decode(
        ckv=32, kpe=8, num_heads=4, seq_len=6, page_size=4,
    )
    assert attn.mla_kernel_available() is False


def test_sdpa_plan_leaves_no_wrapper_behind():
    """``run_mla`` picks its path with ``state.wrapper is not None``, so the SDPA
    plan must leave that slot empty — a paged wrapper cannot read the latent
    cache."""
    resources = _resources(32, 8, 4, 0.1, page_size=4)
    attn = resources["attn"]
    try:
        ingest(resources, "r0")
        assert attn.mla_kernel_available() is False
        with step(resources, {"r0": 4}):
            state = attn._current_plan_states["main"]
            assert state.wrapper is None, "SDPA plan left a stale wrapper behind"
            assert state.sdpa is not None
            assert [r.seq_len for r in state.sdpa.requests] == [4]
    finally:
        cleanup(resources)
