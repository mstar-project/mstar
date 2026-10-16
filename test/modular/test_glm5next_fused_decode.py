"""Fused glm5_next kernels (fused_decode, and the KDA decode in kda_triton) vs the torch
reference — parity on CUDA."""
import pytest
import torch

from mstar.engine.resources.linear_attn.kda import KDAPlan
from mstar.engine.resources.linear_attn.kda_triton import TritonKDAKernels
from mstar.model.components.norm import RMSNorm
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.attention import Glm5NextKdaAttention
from mstar.model.glm5_next.components.decoder_layer import Glm5NextDecoderLayer
from mstar.model.glm5_next.components.moe import Glm5NextMoEGate
from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.kda import Glm5NextKdaConfig, TorchKDAKernels
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
    kda.pool, kda.kda = _Pool(rec, conv), _Kda(plan, kernels)


@pytest.mark.parametrize("batch", [1, 5, 32])
def test_kda_decode_matches_reference(batch):
    heads, head_dim, hidden, slots = 8, 128, 4096, 40
    kda = Glm5NextKdaAttention(Glm5NextKdaConfig(
        hidden_size=hidden, linear_num_heads=heads, linear_head_dim=head_dim),
        dtype=torch.bfloat16).cuda()
    _randomize(kda, 0)
    kda.process_weights_after_loading("cuda")
    # the pool is V-first, [H, V, K]; the reference step is K-first
    rec = torch.randn(slots, heads, head_dim, head_dim, device="cuda") * 0.1
    conv = torch.randn(slots, 3 * heads * head_dim, 3, device="cuda").to(torch.bfloat16)
    slot_index = torch.randperm(slots, device="cuda")[:batch]
    _bind(kda, rec, conv, _decode_plan(slot_index), TritonKDAKernels())
    rec_ref, conv_ref = rec.clone(), conv.clone()
    with torch.no_grad():
        for step in range(3):
            x = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
            r, c = rec_ref[slot_index].transpose(-1, -2), conv_ref[slot_index]
            ref = kda.decode_step(x.unsqueeze(1), r, c).squeeze(1)
            rec_ref[slot_index], conv_ref[slot_index] = r.transpose(-1, -2), c
            out = kda.forward_paged(x, 0)
            # The merged in-projection rounds like the three separate ones up to 1 bf16 ulp.
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2, msg=f"step {step}")
    torch.testing.assert_close(conv, conv_ref, rtol=1e-2, atol=1e-2)
    # that in-projection ulp reaches the state: ~4e-3 at the state's scale
    torch.testing.assert_close(rec, rec_ref, rtol=4e-3, atol=4e-3)


# -- mHC ---------------------------------------------------------------------


def _site(hidden, hc):
    site = Glm5NextHyperConnection(hidden_size=hidden, hc_mult=hc).cuda()
    _randomize(site, 1)
    site.finalize_weights()
    return site


@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("hidden,hc", [(4096, 4), (128, 2)])
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
    # The reference computes in bf16, so it only agrees to a few ulp.
    ref = update_streams(residual, h.unsqueeze(0), post, comb)
    torch.testing.assert_close(fused, ref, rtol=3e-2, atol=5e-2)


@pytest.mark.parametrize("tokens", [1, 64])
def test_hc_pre_with_update_matches_reference(tokens, fp32_matmul):
    site, norm = _site(4096, 4), _norm(4096)
    residual, h, post_p, comb_p = _update_inputs(tokens, 4096, 4)
    with torch.no_grad():
        # The fused update rounds once from fp32; the reference update_streams rounds at
        # every bf16 op, so the site is checked on the exactly-updated streams.
        streams_r = _update_exact(residual, h, post_p, comb_p)
        post_r, comb_r, collapsed = site(streams_r)
        normed_r = norm(collapsed.squeeze(0))
        post, comb, normed, streams = site.forward_fused(
            residual, norm, update=(h, post_p, comb_p))
    torch.testing.assert_close(streams, streams_r, rtol=8e-3, atol=1e-2)
    torch.testing.assert_close(post, post_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb, comb_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(normed, normed_r, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 7, 40])
@pytest.mark.parametrize("fn_dtype", [torch.float32, torch.bfloat16])
def test_hc_pre_update_matches_plain_on_its_streams(tokens, fn_dtype):
    """The mix kernel takes its mix over updated streams it never stores, and the finalize
    recomputes and stores them: the site must equal a plain site over those streams."""
    site, norm = _site(4096, 4), _norm(4096)
    site.fn.data = site.fn.data.to(fn_dtype)
    residual, h, post_p, comb_p = _update_inputs(tokens, 4096, 4)
    with torch.no_grad():
        fused = site.forward_fused(residual, norm, update=(h, post_p, comb_p))
        plain = site.forward_fused(fused[3], norm)
    for name, a, b in zip(("post", "comb", "normed"), fused[:3], plain[:3], strict=True):
        assert torch.equal(a, b), name


# -- router ------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 33, 65])
def test_router_matches_reference(tokens, monkeypatch, fp32_matmul):
    gate = Glm5NextMoEGate(4096, 288, 8, routed_scaling_factor=2.5).cuda()
    _randomize(gate, 2)
    gate.finalize_weights()
    x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
    w, ids = gate(x)
    monkeypatch.setattr(fused_decode, "_ENABLED", False)
    w_r, ids_r = gate(x)
    ids, order = ids.sort(dim=-1)
    ids_r, order_r = ids_r.sort(dim=-1)
    torch.testing.assert_close(ids, ids_r, rtol=0, atol=0)
    torch.testing.assert_close(w.gather(1, order), w_r.gather(1, order_r), rtol=1e-5, atol=1e-6)


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
        layer.self_attn.kda.kernels = TorchKDAKernels()
        monkeypatch.setattr(fused_decode, "_ENABLED", False)
        ref = layer(streams)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
    for k, ref_pool in pools.items():
        torch.testing.assert_close(fused_pools[k], ref_pool, rtol=1e-2, atol=1e-2)
