"""The mHC kernels in ``fused_decode``, and a whole decoder layer on the fused kernels, against
the torch reference on CUDA. The KDA kernels' own parity tests are in test_kda_triton.py."""
import pytest
import torch

from mstar.engine.resources.convenience import KDACallable
from mstar.engine.resources.linear_attn.kda import KDAPlan
from mstar.engine.resources.linear_attn.kda_torch import TorchKDAKernels
from mstar.engine.resources.linear_attn.kda_triton import TritonKDAKernels
from mstar.model.components.norm import RMSNorm
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.decoder_layer import Glm5NextDecoderLayer
from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.mhc import Glm5NextHyperConnection, update_streams

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON),
    reason="fused kernels need CUDA + triton",
)


def _randomize(module, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif name.endswith(("A_log", "dt_bias", "base", "e_score_correction_bias")):
                p.normal_(0.0, 0.5)
            elif name.endswith("scale"):
                p.uniform_(0.5, 1.5)
            elif "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.02)


@pytest.fixture
def fp32_matmul():
    """mstar.engine sets TF32 matmuls globally; the fused kernels accumulate in true fp32."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


def _norm(hidden):
    norm = RMSNorm(hidden, eps=1e-5).cuda().to(torch.bfloat16)
    _randomize(norm, 3)
    return norm


# -- KDA decode ------------------------------------------------------------


class _Pool:
    """One layer's pool blocks, as ``RecurrentStatePool.block`` hands them out."""

    def __init__(self, state, conv):
        self.blocks = {"state": state, "conv": conv}

    def block(self, name, layer):
        return self.blocks[name]


class _Kda:
    """The KDA resource's ``run`` on a hand-built plan."""

    speculative_tokens = 0

    def __init__(self, plan, kernels):
        self.plan, self.kernels = plan, kernels

    def current_plan(self, label=None):
        return self.plan

    def run(self, qkv, g, beta, conv, state, params, gate=None, label=None, spec=None):
        return self.kernels.run_paged(qkv, g, beta, self.plan, conv, state, params, gate=gate)


def _decode_plan(slots):
    slots = slots.to(torch.int32)
    n = slots.shape[0]
    return KDAPlan(slot_ids=slots, has_state=torch.ones_like(slots, dtype=torch.bool),
                   spans=(1,) * n, num_tokens=n, is_decode=True)


def _bind(kda, rec, conv, plan, kernels):
    kda.mix = KDACallable(_Pool(rec, conv), _Kda(plan, kernels))


# -- mHC ---------------------------------------------------------------------


def _site(hidden, hc):
    site = Glm5NextHyperConnection(hidden_size=hidden, hc_mult=hc).cuda()
    _randomize(site, 1)
    site.finalize_weights()
    return site


# past fused_decode._PREFILL_TOKENS at GLM-5.3's width the site takes the prefill kernels
@pytest.mark.parametrize("tokens,hidden,hc", [
    (1, 4096, 4), (7, 4096, 4), (64, 4096, 4), (257, 4096, 4), (1036, 4096, 4),
    (1, 128, 2), (7, 128, 2), (64, 128, 2),
])
def test_hc_pre_matches_reference(tokens, hidden, hc, fp32_matmul):
    site, norm = _site(hidden, hc), _norm(hidden)
    streams = torch.randn(1, tokens, hc, hidden, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        post_r, comb_r, collapsed = site(streams)
        normed_r = norm(collapsed.squeeze(0))
        post, comb, normed, same = site.forward_fused(streams, norm)
    assert same.data_ptr() == streams.data_ptr()
    torch.testing.assert_close(post, post_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb, comb_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(normed, normed_r, rtol=2e-2, atol=2e-2)


def _update_inputs(tokens, hidden, hc):
    torch.manual_seed(4)
    residual = torch.randn(1, tokens, hc, hidden, device="cuda", dtype=torch.bfloat16)
    h = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    post = torch.rand(1, tokens, hc, device="cuda") * 2
    comb = torch.rand(1, tokens, hc, hc, device="cuda")
    return residual, h, post, comb


def _update_exact(residual, h, post, comb):
    return (post.unsqueeze(-1) * h.float().unsqueeze(0).unsqueeze(-2)
            + comb.transpose(-1, -2) @ residual.float()).to(torch.bfloat16)


@pytest.mark.parametrize("tokens", [1, 64])
def test_update_streams_is_fp32_exact(tokens, fp32_matmul):
    residual, h, post, comb = _update_inputs(tokens, 4096, 4)
    exact = _update_exact(residual, h, post, comb)
    fused = fused_decode.update_streams(residual, h, post, comb)
    torch.testing.assert_close(fused, exact, rtol=8e-3, atol=1e-2)
    # the reference computes in bf16, so it only agrees to a few ulp
    ref = update_streams(residual, h.unsqueeze(0), post, comb)
    torch.testing.assert_close(fused, ref, rtol=3e-2, atol=5e-2)


@pytest.mark.parametrize("tokens", [1, 7, 40, 1036])
@pytest.mark.parametrize("fn_dtype", [torch.float32, torch.bfloat16])
def test_hc_pre_with_update_matches_plain_on_its_streams(tokens, fn_dtype, fp32_matmul):
    """With the previous sublayer's update fused in, the site stores the exactly-updated
    streams and gives the bits a plain site gives over them. At decode sizes the mix reads
    updated streams it never stores; at prefill sizes the update is its own pass."""
    site, norm = _site(4096, 4), _norm(4096)
    site.fn.data = site.fn.data.to(fn_dtype)
    residual, h, post_p, comb_p = _update_inputs(tokens, 4096, 4)
    with torch.no_grad():
        fused = site.forward_fused(residual, norm, update=(h, post_p, comb_p))
        plain = site.forward_fused(fused[3], norm)
    torch.testing.assert_close(fused[3], _update_exact(residual, h, post_p, comb_p),
                               rtol=8e-3, atol=1e-2)
    for name, a, b in zip(("post", "comb", "normed"), fused[:3], plain[:3], strict=True):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("tokens", [1, 7, 32])
def test_mhc_chain_with_pdl_is_bitwise_unchanged(tokens, monkeypatch):
    """A layer's decode mHC chain (site, site with the sublayer's update, closing update)
    as dependent launches inside a CUDA graph gives the same bits as without them."""
    if not torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("programmatic dependent launch needs sm90+")
    attn, ffn, norm = _site(4096, 4), _site(4096, 4), _norm(4096)
    residual, h_attn, _, _ = _update_inputs(tokens, 4096, 4)
    h_mlp = torch.randn_like(h_attn)

    def chain():
        post, comb, _, streams = attn.forward_fused(residual, norm)
        post, comb, normed, streams = ffn.forward_fused(streams, norm, update=(h_attn, post, comb))
        return normed, fused_decode.update_streams(streams, h_mlp, post, comb)

    dev = residual.device
    with torch.no_grad():
        monkeypatch.setitem(fused_decode._PDL, dev, False)
        ref = chain()
        monkeypatch.setitem(fused_decode._PDL, dev, True)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            chain()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = chain()
        graph.replay()
        torch.cuda.synchronize()
    for name, a, b in zip(("normed", "streams"), out, ref, strict=True):
        assert torch.equal(a, b), name


# -- whole layer (reduced dims) ------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3])
def test_kda_layer_fused_matches_reference(batch, monkeypatch):
    """The whole layer, fused kernels against the torch reference bundle and mHC."""
    cfg = Glm5NextModelConfig.reduced()
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = Glm5NextDecoderLayer(cfg, 4).cuda()
    finally:
        torch.set_default_dtype(torch.float32)
    _randomize(layer, 5)
    for m in layer.modules():
        if m is not layer and hasattr(m, "process_weights_after_loading"):
            m.process_weights_after_loading("cuda")
    heads, dim = cfg.linear_num_heads, cfg.linear_head_dim
    pools = {
        "state": torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.1,
        "conv": torch.randn(batch + 1, 3 * heads * dim, 3, device="cuda").to(torch.bfloat16),
    }
    plan = _decode_plan(torch.arange(1, batch + 1, device="cuda"))
    _bind(layer.self_attn, pools["state"], pools["conv"], plan, TritonKDAKernels())
    streams = torch.randn(1, batch, cfg.hc_mult, cfg.hidden_size, device="cuda",
                          dtype=torch.bfloat16)
    saved = {k: v.clone() for k, v in pools.items()}
    with torch.no_grad():
        out = layer(streams)
        fused_pools = {k: v.clone() for k, v in pools.items()}
        for k, v in saved.items():
            pools[k].copy_(v)
        layer.self_attn.mix.attn.kernels = TorchKDAKernels()
        monkeypatch.setattr(fused_decode, "_ENABLED", False)
        ref = layer(streams)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
    for k, ref_pool in pools.items():
        torch.testing.assert_close(fused_pools[k], ref_pool, rtol=1e-2, atol=1e-2)
