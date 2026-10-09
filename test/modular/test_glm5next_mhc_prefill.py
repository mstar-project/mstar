"""The mHC site at prefill sizes (fused_decode.hc_pre above _PREFILL_TOKENS: the update as its
own pass, wider mix tiles) vs the torch reference — CUDA + triton. Same tolerances as the
decode-size tests in test_glm5next_fused_decode.py."""
import pytest
import torch

from mstar.model.components.norm import RMSNorm
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.mhc import Glm5NextHyperConnection

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON),
    reason="fused kernels need CUDA + triton",
)

HIDDEN, HC = 4096, 4


@pytest.fixture
def fp32_matmul():
    """mstar.engine sets TF32 matmuls globally; the fused kernels accumulate in true fp32."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


def _site_and_norm(fn_dtype=torch.float32):
    torch.manual_seed(1)
    site = Glm5NextHyperConnection(hidden_size=HIDDEN, hc_mult=HC).cuda()
    norm = RMSNorm(HIDDEN, eps=1e-5).cuda().to(torch.bfloat16)
    with torch.no_grad():
        site.fn.normal_(0.0, 0.02)
        site.base.normal_(0.0, 0.5)
        site.scale.uniform_(0.5, 1.5)
        norm.weight.normal_(1.0, 0.1)
    site.fn.data = site.fn.data.to(fn_dtype)
    site.finalize_weights()
    return site, norm


def _update_inputs(tokens):
    torch.manual_seed(4)
    residual = torch.randn(1, tokens, HC, HIDDEN, device="cuda", dtype=torch.bfloat16)
    h = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    post = torch.rand(1, tokens, HC, device="cuda") * 2
    comb = torch.rand(1, tokens, HC, HC, device="cuda")
    return residual, h, post, comb


@pytest.mark.parametrize("tokens", [257, 1036])
def test_prefill_site_matches_reference(tokens, fp32_matmul):
    assert tokens > fused_decode._PREFILL_TOKENS
    site, norm = _site_and_norm()
    streams = torch.randn(1, tokens, HC, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        post_r, comb_r, collapsed = site(streams)
        normed_r = norm(collapsed.squeeze(0))
        post, comb, normed, same = site.forward_fused(streams, norm)
    assert same.data_ptr() == streams.data_ptr()
    torch.testing.assert_close(post, post_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb, comb_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(normed, normed_r, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("fn_dtype", [torch.float32, torch.bfloat16])
def test_prefill_update_is_its_own_pass(fn_dtype):
    """Above the threshold the site updates with update_streams, then mixes the stored
    streams: it must equal update_streams followed by a plain site, bit for bit."""
    site, norm = _site_and_norm(fn_dtype)
    residual, h, post_p, comb_p = _update_inputs(1036)
    with torch.no_grad():
        fused = site.forward_fused(residual, norm, update=(h, post_p, comb_p))
        streams = fused_decode.update_streams(residual, h, post_p, comb_p)
        plain = site.forward_fused(streams, norm)
    assert torch.equal(fused[3], streams)
    for name, a, b in zip(("post", "comb", "normed"), fused[:3], plain[:3], strict=True):
        assert torch.equal(a, b), name


def test_prefill_update_matches_reference(fp32_matmul):
    """The updated streams against the exact update (a few bf16 ulp apart, as at decode
    sizes), and the site against the reference site on the streams the kernel produced:
    over 1036 tokens, reference streams one ulp off would move post past 1e-4."""
    site, norm = _site_and_norm()
    residual, h, post_p, comb_p = _update_inputs(1036)
    with torch.no_grad():
        exact = (post_p.unsqueeze(-1) * h.float().unsqueeze(0).unsqueeze(-2)
                 + comb_p.transpose(-1, -2) @ residual.float()).to(torch.bfloat16)
        post, comb, normed, streams = site.forward_fused(
            residual, norm, update=(h, post_p, comb_p))
        post_r, comb_r, collapsed = site(streams)
        normed_r = norm(collapsed.squeeze(0))
    torch.testing.assert_close(streams, exact, rtol=8e-3, atol=1e-2)
    torch.testing.assert_close(post, post_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb, comb_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(normed, normed_r, rtol=2e-2, atol=2e-2)
