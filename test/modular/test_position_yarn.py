"""``PositionConfig.yarn``: the YaRN cos/sin table against the checkpoints' own RoPE code, transcribed:
Kimi K2.7-Code's ``DeepseekV3YarnRotaryEmbedding`` (``modeling_deepseek.py``, HF 74797c9c623) and
DeepSeek-V4.1's ``precompute_freqs_cis`` (``inference/model.py``, 314113f8a7)."""

import dataclasses
import math

import pytest
import torch

from mstar.engine.resources.position.config import PositionConfig, YarnScaling, yarn_mscale
from mstar.engine.resources.position.manager import RopeManager

K27 = YarnScaling(factor=64.0, original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
                  attention_factor=yarn_mscale(64.0, 1.0) / yarn_mscale(64.0, 1.0))
V41 = YarnScaling(factor=16.0, original_max_position_embeddings=65536)


def _k27_cos_sin(dim, base, seq_len, factor, orig, beta_fast, beta_slow, mscale, mscale_all_dim):
    """DeepseekV3YarnRotaryEmbedding._set_cos_sin_cache, in fp32."""
    def correction_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    mask = 1.0 - ramp
    inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    m = yarn_mscale(factor, mscale) / yarn_mscale(factor, mscale_all_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * m, emb.sin() * m


def _v41_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """DeepSeek-V4.1's precompute_freqs_cis."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def corrected_dim(rotations):
        return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corrected_dim(beta_fast)), 0)
    high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    smooth = 1 - ramp
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def manager(yarn, theta, dim=64, seq=512, table=torch.float32, device="cpu", **kw):
    config = PositionConfig(kv_cache="kv", interleave=True, rope_theta=theta, rotary_dim=dim,
                            rope_table_dtype=table, yarn=yarn, **kw)
    return RopeManager(config, torch.device(device), max_seq_len=seq, head_dim=dim)


def test_kimi_k27_table_is_its_reference_rotary_embedding():
    cos, sin = _k27_cos_sin(64, 50000.0, 512, 64.0, 4096, 32.0, 1.0, 1.0, 1.0)
    table = manager(K27, 50000.0)._rope_table
    assert torch.equal(table, torch.cat([cos[:, :32], sin[:, :32]], dim=-1))
    # the checkpoint rounds its table to bf16, as rope_table_dtype=bfloat16 does
    bf16 = manager(K27, 50000.0, table=torch.bfloat16)._rope_table
    assert torch.equal(bf16, torch.cat([cos[:, :32], sin[:, :32]], dim=-1).bfloat16().float())


def test_an_attention_factor_scales_the_table():
    m = yarn_mscale(16.0)
    cos, sin = _k27_cos_sin(64, 1e4, 512, 16.0, 65536, 32.0, 1.0, 1.0, 0.0)
    assert m == pytest.approx(0.1 * math.log(16) + 1)
    table = manager(dataclasses.replace(V41, attention_factor=m), 1e4)._rope_table
    assert torch.equal(table, torch.cat([cos[:, :32], sin[:, :32]], dim=-1))


def test_deepseek_v41_compressed_layer_table():
    cis = _v41_freqs_cis(64, 512, 65536, 160000.0, 16.0, 32.0, 1.0)
    table = manager(V41, 160000.0)._rope_table
    # V4.1 divides the stretched frequencies after inverting (freqs / factor), main's form inverts
    # factor * theta^x: the frequencies can differ in the last fp32 bit
    torch.testing.assert_close(table, torch.cat([cis.real, cis.imag], dim=-1), rtol=0, atol=2e-5)


def test_softmax_scale_of_kimi_k27():
    # qk_head_dim ** -0.5 * mscale(factor, mscale_all_dim) ** 2, as modeling_deepseek.py sets it
    assert 192 ** -0.5 * yarn_mscale(64.0, 1.0) ** 2 == pytest.approx(0.14468, abs=1e-5)


def test_refusals(monkeypatch):
    with pytest.raises(ValueError, match="rope_table_dtype"):
        manager(K27, 50000.0, table=None, device="cuda")
    with pytest.raises(ValueError, match="rope_scale"):
        manager(K27, 50000.0, rope_scale=2.0)
    with pytest.raises(ValueError, match="Llama 3.1"):
        manager(K27, 50000.0, low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192)
    assert manager(K27, 50000.0).fusable_rope_theta(64) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer's RoPE kernel needs CUDA")
def test_cuda_rotates_interleaved_pairs_by_the_yarn_angles():
    """The kernel's rotation from the YaRN table equals rotating each (even, odd) pair by the
    reference's angle, which is K2.7's de-interleaved rotate_half up to the same permutation of
    q and k."""
    seq, dim, heads = 4096, 64, 4
    rope = manager(K27, 50000.0, seq=seq, device="cuda")
    pos = torch.randint(0, seq, (32,), device="cuda", dtype=torch.int32)
    q = torch.randn(32, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(32, 1, dim, device="cuda", dtype=torch.bfloat16)
    got_q, got_k = rope.apply_qk(q.clone(), k.clone(), pos_ids=pos)
    cos, sin = _k27_cos_sin(dim, 50000.0, seq, 64.0, 4096, 32.0, 1.0, 1.0, 1.0)
    c, s = cos[:, : dim // 2].cuda()[pos.long()], sin[:, : dim // 2].cuda()[pos.long()]

    def rotate(x):
        x = x.float()
        even, odd = x[..., 0::2], x[..., 1::2]
        cc, ss = c[:, None], s[:, None]
        return torch.stack([even * cc - odd * ss, even * ss + odd * cc], dim=-1).flatten(-2)

    torch.testing.assert_close(got_q.float(), rotate(q), rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(got_k.float(), rotate(k), rtol=1e-2, atol=2e-2)
