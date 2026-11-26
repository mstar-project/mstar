"""Unit tests for DeepSeek-V4.1 config, RoPE/YaRN, MX rounding, ReferenceRMSNorm, and Engram hashing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from sympy import isprime

from mstar.model.components.deepseek_v41.config import DeepSeekV41Config
from mstar.model.components.deepseek_v41.engram_hash import (
    EngramLayout,
    bucket_primes,
    build_compressed_token_map,
    compute_engram_hashes,
    compute_hash_multipliers,
)
from mstar.model.components.deepseek_v41.quant import (
    ReferenceRMSNorm,
    dequant_block_fp8,
    pack_e2m1,
    round_fp4,
    round_fp8,
    unpack_e2m1,
)
from mstar.model.components.deepseek_v41.rope import (
    DeepSeekV41Rope,
    apply_rotary_emb,
    inv_freq,
    precompute_freqs_cis,
    rotate_tail,
)


def _pinned_config_dict() -> dict:
    return json.loads((Path(__file__).parent / "data/deepseek_v41_pinned_config.json").read_text())


def test_from_hf_matches_pinned_config_and_schedules() -> None:
    raw = _pinned_config_dict()
    cfg = DeepSeekV41Config.from_hf(raw)
    assert cfg.num_hidden_layers == 40
    assert cfg.hidden_size == 5120
    assert cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 1
    assert cfg.head_dim == 512
    assert cfg.qk_rope_head_dim == 64
    assert cfg.q_lora_rank == 1280
    assert cfg.o_lora_rank == 1024
    assert cfg.o_groups == 8
    assert cfg.sliding_window == 128
    assert cfg.kv_source_layer_ids == (2, 8, 14, 20)
    assert cfg.index_source_layer_ids == (2, 8, 14, 20, 24, 28, 32, 36)
    assert cfg.candidate_source_layer_id == 20
    assert cfg.engram_layer_ids == (1, 14)
    assert cfg.engram_num_embeddings == (384006168, 384016682)
    assert cfg.kv_source_of(0) is None
    assert cfg.kv_source_of(2) == 2
    assert cfg.kv_source_of(7) == 2
    assert cfg.kv_source_of(25) == 20
    assert cfg.index_source_of(13) == 8
    assert cfg.index_source_of(39) == 36
    assert not cfg.uses_candidates(20)
    assert cfg.uses_candidates(24)
    assert cfg.pooled_source_layer_ids == (2, 8, 14)


def test_config_rejects_invalid_formats() -> None:
    raw = _pinned_config_dict()
    bad_quant = json.loads(json.dumps(raw))
    bad_quant["quantization_config"]["expert_dtype"] = "bf16"
    with pytest.raises(ValueError, match="quantization_config"):
        DeepSeekV41Config.from_hf(bad_quant)

    bad_rope = json.loads(json.dumps(raw))
    bad_rope["text_config"]["rope_scaling"]["rope_type"] = "linear"
    with pytest.raises(ValueError, match="rope_scaling"):
        DeepSeekV41Config.from_hf(bad_rope)

    with pytest.raises(ValueError, match="kv_source_layer_id"):
        DeepSeekV41Config.reduced(kv_source_layer_ids=(0, 1))


def test_rope_plain_yarn_and_inverse_roundtrip() -> None:
    cfg = DeepSeekV41Config.from_hf(_pinned_config_dict())
    rope = DeepSeekV41Rope(cfg)
    pos = torch.tensor([0, 7, 1025], dtype=torch.long)
    plain = rope.angles(pos, compressed=False)
    comp = rope.angles(pos, compressed=True)
    assert plain.cos.shape == (3, cfg.qk_rope_head_dim // 2)
    assert comp.cos.shape == (3, cfg.qk_rope_head_dim // 2)

    # Check YaRN low/high band against formula: highest frequency unchanged, lowest scaled by 1/factor.
    plain_freqs = inv_freq(cfg.qk_rope_head_dim, cfg.compress_rope_theta)
    yarn_freqs = inv_freq(
        cfg.qk_rope_head_dim,
        cfg.compress_rope_theta,
        cfg.rope_original_max_position_embeddings,
        cfg.rope_factor,
        cfg.rope_beta_fast,
        cfg.rope_beta_slow,
    )
    assert torch.isclose(yarn_freqs[0], plain_freqs[0])
    assert torch.isclose(yarn_freqs[-1], plain_freqs[-1] / cfg.rope_factor)

    # Exact inverse RoPE roundtrip on last qk_rope_head_dim dims, leaving prefix untouched.
    gen = torch.Generator().manual_seed(401)
    x = torch.randn(3, 4, cfg.head_dim, generator=gen, dtype=torch.float32)
    rot = rotate_tail(x, comp)
    assert torch.equal(rot[..., : -cfg.qk_rope_head_dim], x[..., : -cfg.qk_rope_head_dim])
    assert not torch.allclose(rot[..., -cfg.qk_rope_head_dim :], x[..., -cfg.qk_rope_head_dim :])
    restored = rotate_tail(rot, comp, inverse=True)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-5)

    # precompute_freqs_cis matches DeepSeekV41Rope
    direct = precompute_freqs_cis(pos, cfg.qk_rope_head_dim, cfg.rope_theta)
    torch.testing.assert_close(direct.cos, plain.cos)
    torch.testing.assert_close(direct.sin, plain.sin)
    # 2D input apply_rotary_emb
    v2 = torch.randn(3, cfg.qk_rope_head_dim, generator=gen, dtype=torch.float32)
    rt = apply_rotary_emb(apply_rotary_emb(v2, plain), plain, inverse=True)
    torch.testing.assert_close(rt, v2, atol=1e-5, rtol=1e-5)


def test_rope_angles_compile_without_recompile_and_survive_meta_to_empty(dynamo_counts) -> None:
    cfg = DeepSeekV41Config.reduced()
    ref = DeepSeekV41Rope(cfg)
    pos = torch.tensor([0, 3, 7], dtype=torch.long)

    with torch.device("meta"):
        rope = DeepSeekV41Rope(cfg).to(torch.bfloat16)
    assert rope.state_dict() == {}
    rope.to_empty(device="cpu")
    assert rope.plain_freqs.dtype == torch.float32
    assert rope.compress_freqs.dtype == torch.float32
    assert torch.equal(rope.plain_freqs, ref.plain_freqs)
    assert torch.equal(rope.compress_freqs, ref.compress_freqs)

    for compressed in (False, True):
        counts = dynamo_counts(lambda p, c=compressed: rope.angles(p, c), pos)
        assert counts.compiles == 1
        assert counts.graph_calls == 1


def test_mx_rounding_and_reference_rmsnorm_bf16_fidelity() -> None:
    gen = torch.Generator().manual_seed(42)
    x = torch.randn((4, 64), generator=gen, dtype=torch.bfloat16) * 3.5
    x8 = round_fp8(x, block=32)
    assert x8.dtype == torch.bfloat16
    assert torch.isfinite(x8).all()
    # Idempotence on already-rounded grid values for power-of-two blocks
    torch.testing.assert_close(round_fp8(x8, block=32), x8, rtol=0, atol=0)

    x4_e8 = round_fp4(x, block=32, e4m3_scale=False)
    x4_e4 = round_fp4(x, block=16, e4m3_scale=True)
    assert x4_e8.dtype == torch.bfloat16 and x4_e4.dtype == torch.bfloat16
    # E2M1 pack/unpack round-trip
    grid_vals = torch.tensor([[-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]])
    torch.testing.assert_close(unpack_e2m1(pack_e2m1(grid_vals)), grid_vals, rtol=0, atol=0)

    # dequant_block_fp8 exactness
    w = torch.randint(-8, 8, (32, 32), generator=gen, dtype=torch.int8).to(torch.float8_e4m3fn)
    s = torch.full((1, 1), 0.25, dtype=torch.float8_e8m0fnu)
    dq = dequant_block_fp8(w, s, torch.bfloat16)
    torch.testing.assert_close(dq, (w.float() * 0.25).to(torch.bfloat16), rtol=0, atol=0)

    # ReferenceRMSNorm keeps fp32 multiply before casting back to BF16
    norm = ReferenceRMSNorm(64, eps=1e-20)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 64))
    y = norm(x)
    expected = (norm.weight.float() * (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20))).to(
        torch.bfloat16
    )
    assert torch.equal(y, expected)


def test_engram_primes_match_pinned_config_and_token_map_normalizes() -> None:
    cfg = DeepSeekV41Config.from_hf(_pinned_config_dict())
    primes = bucket_primes(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_vocab_size, cfg.engram_n_heads)
    flat_all = [p for layer in primes for per_n in layer for p in per_n]
    assert len(flat_all) == len(set(flat_all))
    assert all(isprime(p) for p in flat_all)
    assert tuple(sum(p for per_n in layer for p in per_n) for layer in primes) == cfg.engram_num_embeddings

    mult = compute_hash_multipliers(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_compressed_vocab_size)
    assert mult.shape == (2, 4)
    assert mult.dtype == torch.int64
    assert bool((mult % 2 == 1).all())

    # Synthetic fast tokenizer verifying NFKC/strip-accents/lowercase/whitespace collapse & single-space sentinel.
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "Café": 1, "cafe": 2, " ": 3, "\t ": 4, "a  b": 5, "a b": 6, " a ": 7, "a": 8}
    tok_backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=tok_backend, pad_token="<pad>")
    tmap = build_compressed_token_map(fast_tok)
    assert tmap[1] == tmap[2]  # Café and cafe fold together
    assert tmap[5] == tmap[6]  # "a  b" and "a b" collapse
    assert tmap[3] == tmap[4]  # whitespace runs collapse to single space preserved by sentinel
    assert tmap[7] == tmap[8]  # surrounding whitespace strips around words
    assert tmap[3] != tmap[8]  # single space stays distinct from word tokens


def _scalar_engram_reference(
    ids: list[int],
    mask: list[bool],
    layout: EngramLayout,
) -> torch.Tensor:
    tmap = layout.token_map.tolist()
    mult = layout.multipliers.tolist()
    primes = layout.primes.tolist()
    offsets = layout.offsets.tolist()
    n = layout.max_ngram_size
    comp = [tmap[tid] if m else -1 for tid, m in zip(ids, mask, strict=True)]
    out = []
    for t in range(len(ids)):
        win = []
        blocked = False
        for shift in range(n):
            idx = t - shift
            if idx < 0 or comp[idx] < 0:
                blocked = True
            win.append(layout.pad_id if blocked else comp[idx])
        layer_rows = []
        for l_idx in range(len(primes)):
            cols = []
            col_i = 0
            for ng in range(1, n):
                rolling = win[0] * mult[l_idx][0]
                for s in range(1, ng + 1):
                    rolling ^= win[s] * mult[l_idx][s]
                for h_p in primes[l_idx][ng - 1]:
                    cols.append((rolling % h_p) + offsets[l_idx][col_i])
                    col_i += 1
            layer_rows.append(cols)
        out.append(layer_rows)
    return torch.tensor(out, dtype=torch.long)


def test_engram_hashing_dead_masking_and_batch_isolation() -> None:
    cfg = DeepSeekV41Config.reduced(engram_layer_ids=(1, 4), engram_num_embeddings=(0, 0))
    # Non-trivial token map with folding
    tmap = np.arange(cfg.vocab_size, dtype=np.int64) % 100
    layout = EngramLayout.from_config(cfg, tmap)

    seq_a = [10, 125, 30, 42, 7]
    mask_a = [True, True, False, True, True]
    seq_b = [99, 11, 22, 33]
    mask_b = [True, True, True, True]

    ref_a = _scalar_engram_reference(seq_a, mask_a, layout)
    ref_b = _scalar_engram_reference(seq_b, mask_b, layout)

    # 2D batch check
    ids_a = torch.tensor([seq_a])
    m_a = torch.tensor([mask_a])
    h_a, _ = compute_engram_hashes(ids_a, layout, token_mask=m_a)
    assert torch.equal(h_a[0], ref_a)

    # Step-by-step chunked / decode packed check with TWO interleaved requests (no leakage!)
    h_a_p1, tail_a = compute_engram_hashes(ids_a[:, :3], layout, token_mask=m_a[:, :3])
    h_a_p2, tail_a = compute_engram_hashes(ids_a[:, 3:], layout, prev_tail=tail_a, token_mask=m_a[:, 3:])
    assert torch.equal(torch.cat([h_a_p1, h_a_p2], dim=1)[0], ref_a)

    # Packed 1D batch: prefill request A (len 3) + request B (len 2), then decode/chunk rest
    packed_ids = torch.tensor(seq_a[:3] + seq_b[:2], dtype=torch.long)
    packed_mask = torch.tensor(mask_a[:3] + mask_b[:2], dtype=torch.bool)
    pos = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    row_req = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    req_first = torch.tensor([0, 0, 0, 3, 3], dtype=torch.long)
    h_p1, packed_tail = compute_engram_hashes(
        packed_ids, layout, pos=pos, row_req=row_req, req_first=req_first, token_mask=packed_mask
    )
    assert torch.equal(h_p1[:3], ref_a[:3])
    assert torch.equal(h_p1[3:], ref_b[:2])

    # Second packed step (2 tokens for A, 2 tokens for B)
    packed_ids2 = torch.tensor(seq_a[3:] + seq_b[2:], dtype=torch.long)
    packed_mask2 = torch.tensor(mask_a[3:] + mask_b[2:], dtype=torch.bool)
    pos2 = torch.tensor([3, 4, 2, 3], dtype=torch.long)
    row_req2 = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    req_first2 = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    h_p2, _ = compute_engram_hashes(
        packed_ids2,
        layout,
        pos=pos2,
        row_req=row_req2,
        req_first=req_first2,
        prev_tail=packed_tail,
        token_mask=packed_mask2,
    )
    assert torch.equal(h_p2[:2], ref_a[3:])
    assert torch.equal(h_p2[2:], ref_b[2:])
    assert math.isfinite(float(h_p2.sum().item()))
