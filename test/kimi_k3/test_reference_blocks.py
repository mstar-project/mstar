"""Reference-block parity against the Hugging Face modeling code (CPU, no kernels).

Everything here runs on CPU with random weights from the tiny config; it checks that the
pure-PyTorch references in ``mstar.model.kimi_k3.reference`` reproduce the checkpoint's
own modeling classes bit-for-bit (fp32) or within bf16 tolerance.
"""

import torch

from mstar.model.kimi_k3.config import KimiK3Config
from mstar.model.kimi_k3.reference.attn_res import attn_res_read, attn_res_score_weight
from mstar.model.kimi_k3.reference.mla import MLAWeights, mla_forward_absorbed, mla_forward_dense
from mstar.model.kimi_k3.reference.moe import LatentMoEWeights, latent_moe_forward
from mstar.model.kimi_k3.reference.mxfp4 import dequant_mxfp4, pack_fp4_codes, quant_mxfp4, unpack_fp4_codes
from mstar.model.kimi_k3.reference.router import noaux_tc_route
from mstar.model.kimi_k3.reference.situ import situ_and_mul

torch.manual_seed(0)


def test_config_parses_full_and_tiny(tiny_dir):
    full_dir = tiny_dir.parent / "hf_kimi_k3"
    if (full_dir / "config.json").exists():
        cfg = KimiK3Config.from_hf_dir(full_dir)
        t = cfg.text
        assert t.num_hidden_layers == 93 and t.num_kda_layers == 69 and t.num_mla_layers == 24
        assert sorted(t.full_attn_layers)[:3] == [3, 7, 11] and 92 in t.full_attn_layers
        assert not t.is_moe_layer(0) and t.is_moe_layer(1) and t.is_moe_layer(92)
        assert t.num_attn_res_blocks == 8 and t.attn_res_blocks_before(0) == 0
        assert t.attn_res_blocks_before(12) == 1 and t.attn_res_blocks_before(13) == 2
        assert t.attn_res_blocks_before(92) == 8
        assert t.mla_kv_latent_dim == 576 and t.kda_projection_size == 12288
        assert cfg.quant is not None and cfg.quant.is_mxfp4 and cfg.quant.group_size == 32
        assert cfg.stop_token_id == 163586
    cfg = KimiK3Config.from_hf_dir(tiny_dir)
    t = cfg.text
    assert t.num_hidden_layers == 8 and sorted(t.full_attn_layers) == [3, 7]
    assert sorted(t.kda_layers) == [0, 1, 2, 4, 5, 6]
    assert t.attn_res_block_size == 4 and t.num_attn_res_blocks == 2


def test_situ_matches_hf(hf_modeling):
    x = torch.randn(4, 7, 2 * 96, dtype=torch.bfloat16) * 8
    ref = hf_modeling.SituAndMul(beta=4.0, linear_beta=25.0)(x)
    out = situ_and_mul(x, 4.0, 25.0)
    assert torch.equal(out, ref)
    ref = hf_modeling.SituAndMul(beta=1.0, linear_beta=None)(x.float())
    assert torch.equal(situ_and_mul(x.float(), 1.0, None), ref)


def test_attn_res_matches_hf(hf_modeling):
    d, t = 64, 5
    for m in (0, 1, 3, 8):
        prefix = torch.randn(t, d)
        blocks = torch.randn(t, m, d)
        proj = torch.nn.Linear(d, 1, bias=False)
        norm = hf_modeling.KimiRMSNorm(d, eps=1e-5)
        with torch.no_grad():
            norm.weight.normal_()
        if m == 0:
            ref = prefix  # the HF layer skips the read when the stack is empty
        else:
            ref = hf_modeling._apply_attn_res(prefix, blocks, proj, norm)
        out = attn_res_read(prefix, blocks, attn_res_score_weight(norm.weight, proj.weight), eps=1e-5)
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
        # bf16 inputs: reference casts the fp32 mixture back to bf16
        out_bf = attn_res_read(prefix.bfloat16(), blocks.bfloat16(), attn_res_score_weight(norm.weight, proj.weight))
        if m:
            ref_bf = hf_modeling._apply_attn_res(prefix.bfloat16(), blocks.bfloat16(), proj, norm)
            assert torch.equal(out_bf, ref_bf)


def test_router_matches_hf(hf_modeling, hf_text_config):
    gate = hf_modeling.KimiMoEGate(hf_text_config).eval()
    with torch.no_grad():
        gate.e_score_correction_bias.normal_(std=0.1)
    x = torch.randn(2, 5, hf_text_config.hidden_size, dtype=torch.bfloat16)
    ref_idx, ref_w = gate(x)
    idx, w = noaux_tc_route(
        x, gate.weight, gate.e_score_correction_bias, hf_text_config.num_experts_per_token,
        renormalize=hf_text_config.moe_renormalize,
        routed_scaling_factor=hf_text_config.routed_scaling_factor,
    )
    # order within top-k is unspecified (sorted=False); compare as sets per token
    assert torch.equal(idx.sort(-1).values, ref_idx.sort(-1).values)
    torch.testing.assert_close(w.sort(-1).values, ref_w.sort(-1).values)


def test_mxfp4_roundtrip_and_layout():
    codes = torch.randint(0, 16, (6, 64), dtype=torch.uint8)
    packed = pack_fp4_codes(codes)
    assert packed.shape == (6, 32) and torch.equal(unpack_fp4_codes(packed), codes)
    # low nibble is element 2j
    assert int(packed[0, 0]) == int(codes[0, 0]) | (int(codes[0, 1]) << 4)
    w = torch.randn(8, 128) * 3
    p, s = quant_mxfp4(w)
    assert p.shape == (8, 64) and s.shape == (8, 4) and p.dtype == torch.uint8 and s.dtype == torch.uint8
    deq = dequant_mxfp4(p, s, dtype=torch.float32)
    # every group's max must be representable (<= 6 * scale) and each value must be the
    # nearest point of the (non-uniform) E2M1 grid at that scale
    scale = torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)
    assert ((deq / scale).abs() <= 6).all()
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])
    nearest = grid[((w / scale).abs()[..., None] - grid).abs().argmin(-1)] * torch.sign(w) * scale
    assert ((deq - nearest).abs() <= 1e-6).all()
    # exactness: values already on the grid round-trip losslessly, in bf16 too
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -3, -6] * 11)[:128].repeat(4, 1) * 4.0
    p2, s2 = quant_mxfp4(grid)
    assert torch.equal(dequant_mxfp4(p2, s2, dtype=torch.bfloat16).float(), grid)


def test_mla_dense_matches_hf_and_absorbed(hf_modeling, hf_text_config):
    torch.manual_seed(1)
    m = hf_modeling.KimiMLAAttention(hf_text_config, layer_idx=3).float().eval()
    for p in m.parameters():
        with torch.no_grad():
            p.normal_(std=0.05)
    t = 9
    x = torch.randn(1, t, hf_text_config.hidden_size)
    mask = torch.full((t, t), float("-inf")).triu(1)[None, None]
    ref = m(x, attention_mask=mask)
    w = MLAWeights.from_hf_module(m)
    out_dense, latent = mla_forward_dense(w, x[0])
    out_abs, latent2 = mla_forward_absorbed(w, x[0])
    torch.testing.assert_close(out_dense, ref[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_abs, ref[0], rtol=1e-4, atol=1e-4)
    assert latent.shape == (t, w.kv_lora_rank + w.qk_rope_head_dim) and torch.equal(latent, latent2)
    # incremental decode against the cache equals the last row of a full forward
    out_step, _ = mla_forward_absorbed(w, x[0, -1:], latent_cache=latent[:-1])
    torch.testing.assert_close(out_step, ref[0, -1:], rtol=1e-4, atol=1e-4)


def test_latent_moe_matches_hf(hf_modeling, hf_text_config):
    torch.manual_seed(2)
    m = hf_modeling.KimiSparseMoeBlock(hf_text_config).float().eval()
    for p in m.parameters():
        with torch.no_grad():
            p.normal_(std=0.05)
    x = torch.randn(1, 6, hf_text_config.hidden_size)
    with torch.no_grad():
        ref = m(x)
        w = LatentMoEWeights.from_hf_module(m)
        out = latent_moe_forward(w, x[0])
    torch.testing.assert_close(out, ref[0], rtol=1e-4, atol=1e-4)
