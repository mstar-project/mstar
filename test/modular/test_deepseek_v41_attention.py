"""CPU parity tests for DeepSeek-V4.1 attention, compressor, CSA2 indexer, candidate selection, and packed steps."""

from __future__ import annotations

import torch

from mstar.model.components.deepseek_v41.attention import (
    DeepSeekV41Attention,
    SharedSelection,
    select_candidate_blocks,
)
from mstar.model.components.deepseek_v41.backbone import DeepSeekV41TextCache
from mstar.model.components.deepseek_v41.config import DeepSeekV41Config
from mstar.model.components.deepseek_v41.quant import round_fp4, round_fp8
from mstar.model.components.deepseek_v41.rope import DeepSeekV41Rope, rotate_tail


def _init_attn(attn: DeepSeekV41Attention, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if "norm" in name:
                p.copy_(0.8 + 0.2 * torch.rand(p.shape, generator=gen, dtype=torch.float32).to(p.dtype))
            elif name == "attn_sink":
                p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.2)
            else:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.dtype) * 0.1)


def _naive_layer_reference(
    attn: DeepSeekV41Attention,
    x_seq: torch.Tensor,  # [S, hidden]
    rope: DeepSeekV41Rope,
    compress_kv: torch.Tensor | None,
    compress_k: torch.Tensor | None,
    shared: SharedSelection,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Independent per-token reference computing SWA + compressed cache + sink softmax + inverse RoPE."""
    seq = x_seq.shape[0]
    pos = torch.arange(seq, dtype=torch.long)
    compressed = attn.ratio > 0
    angles = rope.angles(pos, compressed)
    q_lat, q = attn._query(x_seq, angles)
    kv_win = attn._window_latent(x_seq, angles)

    if compressed:
        if attn.compressor is not None:
            lat, _ = attn.compressor(x_seq.unsqueeze(0), 0)
            assert lat is not None
            n_groups = lat.shape[1]
            g_pos = torch.arange(n_groups, dtype=torch.long) * attn.ratio
            g_ang = rope.angles(g_pos, True)
            compress_kv = round_fp4(rotate_tail(lat[0], g_ang), block=16, e4m3_scale=True)
            if attn.indexer is not None and attn.indexer.owns_keys:
                compress_k = attn.indexer.keys(lat[0], g_ang)
        if attn.indexer is not None:
            iq, iw = attn.indexer.query(x_seq, q_lat, angles)
            assert compress_k is not None
            seen = (pos + 1) // attn.ratio
            dots = torch.einsum("thd,gd->thg", iq.float(), compress_k.float()).relu_()
            scores = (dots * iw.float().unsqueeze(-1)).sum(dim=1)
            g_idx = torch.arange(compress_k.shape[0])[None, :]
            scores = scores.masked_fill(g_idx >= seen[:, None], -torch.inf)
            if attn.is_candidate_source:
                shared.candidates = select_candidate_blocks(
                    scores, seen, attn.candidate_topk_blocks, attn.candidate_block_size
                )
            elif attn.uses_candidates and shared.candidates is not None:
                scores = scores.masked_fill(~shared.candidates[:, : compress_k.shape[0]], -torch.inf)
            vals, idxs = scores.sort(dim=-1, descending=True, stable=True)
            k_pick = min(attn.index_topk, compress_k.shape[0])
            shared.slots = torch.where(vals[:, :k_pick] > -torch.inf, idxs[:, :k_pick], -1)

    outs = []
    for t in range(seq):
        lo = max(0, t - attn.window + 1)
        win_rows = kv_win[lo : t + 1]
        if compressed:
            assert shared.slots is not None and compress_kv is not None
            picked = shared.slots[t]
            valid_picked = picked[picked >= 0]
            all_rows = torch.cat([win_rows, compress_kv[valid_picked]], dim=0)
        else:
            all_rows = win_rows
        logits = torch.einsum("hd,kd->hk", q[t].float(), all_rows.float()) * attn.sm_scale
        probs = torch.cat([logits, attn.attn_sink.float().unsqueeze(-1)], dim=-1).softmax(dim=-1)[:, :-1]
        outs.append(torch.einsum("hk,kd->hd", probs, all_rows.float()).to(q.dtype))

    out_h = torch.stack(outs, dim=0)
    out_rot = rotate_tail(out_h, angles, inverse=True)
    out_r = round_fp8(out_rot.reshape(seq, attn.num_groups, -1), attn.block)
    wo_a = attn.wo_a.weight.view(attn.num_groups, attn.o_lora_rank, -1)
    proj = torch.einsum("sgd,grd->sgr", out_r.float(), wo_a.float()).to(q.dtype)
    return attn.wo_b(proj.flatten(1)), compress_kv, compress_k


def test_attention_prefill_decode_matches_naive_reference_across_all_layer_roles() -> None:
    cfg = DeepSeekV41Config.reduced()
    rope = DeepSeekV41Rope(cfg)
    layers = [DeepSeekV41Attention(cfg, i).eval() for i in range(cfg.num_hidden_layers)]
    for i, layer in enumerate(layers):
        _init_attn(layer, seed=100 + i)

    seq_len = 21
    x = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 0.5

    # Run independent naive reference across all 7 layers
    # (covering ratio=0, ratio=2, ratio=1, candidate_source, uses_candidates).
    ref_outputs = []
    ref_shared = SharedSelection()
    ref_ckv: dict[int, torch.Tensor] = {}
    ref_ck: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for i, layer in enumerate(layers):
            src = cfg.kv_source_of(i)
            out_i, ckv_i, ck_i = _naive_layer_reference(
                layer,
                x[0],
                rope,
                ref_ckv.get(src) if src is not None else None,
                ref_ck.get(src) if src is not None else None,
                ref_shared,
            )
            if src is not None and layer.compressor is not None:
                assert ckv_i is not None and ck_i is not None
                ref_ckv[src] = ckv_i
                ref_ck[src] = ck_i
            ref_outputs.append(out_i)

    # Run DeepSeekV41Attention via prefill (0..15) + chunked prefill (15..18) + token-by-token decode (18..21)
    cache = DeepSeekV41TextCache(cfg, 1, 64, torch.device("cpu"), torch.float32)
    step_outputs: list[list[torch.Tensor]] = [[] for _ in layers]
    with torch.no_grad():
        for start, end in ((0, 15), (15, 18), (18, 19), (19, 20), (20, 21)):
            step_shared = SharedSelection()
            for i, layer in enumerate(layers):
                cache.bind_layer(i)
                step_outputs[i].append(layer(x[:, start:end], start, rope, cache, step_shared)[0])

    for i in range(cfg.num_hidden_layers):
        got = torch.cat(step_outputs[i], dim=0)
        torch.testing.assert_close(got, ref_outputs[i], rtol=1e-4, atol=1e-4)


def test_select_candidate_blocks_pins_newest_partial_block_and_handles_empty() -> None:
    # 10 groups, block_size=4 -> 3 blocks: [0..3], [4..7], [8..9]
    scores = torch.tensor([[10.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, -5.0, -5.0]])
    groups = torch.tensor([10])
    keep = select_candidate_blocks(scores, groups, topk_blocks=2, block_size=4)
    # Block 0 (score 10.0) and newest block 2 (pinned to +inf even though score is -5.0) must be kept; block 1 dropped.
    expected = torch.tensor([[True, True, True, True, False, False, False, False, True, True]])
    assert torch.equal(keep, expected)

    # Zero groups: nothing pinned
    empty_scores = torch.full((1, 4), -torch.inf)
    empty_keep = select_candidate_blocks(empty_scores, torch.tensor([0]), topk_blocks=2, block_size=4)
    assert not empty_keep.any()


def test_forward_packed_isolates_multiple_requests() -> None:
    cfg = DeepSeekV41Config.reduced()
    rope = DeepSeekV41Rope(cfg)
    attn = DeepSeekV41Attention(cfg, layer_id=1).eval()  # ratio=2 compressor + indexer
    _init_attn(attn, seed=7)

    x_a = torch.randn(1, 7, cfg.hidden_size, dtype=torch.float32)
    x_b = torch.randn(1, 5, cfg.hidden_size, dtype=torch.float32)

    cache_a = DeepSeekV41TextCache(cfg, 1, 32, torch.device("cpu"), torch.float32)
    cache_b = DeepSeekV41TextCache(cfg, 1, 32, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        cache_a.bind_layer(1)
        ref_a_1 = attn(x_a[:, :5], 0, rope, cache_a, SharedSelection())[0]
        cache_b.bind_layer(1)
        ref_b_1 = attn(x_b[:, :3], 0, rope, cache_b, SharedSelection())[0]
        cache_a.bind_layer(1)
        ref_a_2 = attn(x_a[:, 5:6], 5, rope, cache_a, SharedSelection())[0]
        cache_b.bind_layer(1)
        ref_b_2 = attn(x_b[:, 3:4], 3, rope, cache_b, SharedSelection())[0]

        # Packed step 1: req 0 (5 toks at pos 0..4), req 1 (3 toks at pos 0..2)
        held_win = torch.zeros(2, cfg.sliding_window, cfg.head_dim)
        ckv = torch.zeros(2, 32, cfg.head_dim)
        ck = torch.zeros(2, 32, cfg.index_head_dim)
        p_kv = torch.zeros(2, 2, cfg.head_dim)
        p_sc = torch.full((2, 2, cfg.head_dim), -torch.inf)
        out1, held_win, ckv, ck, partial = attn.forward_packed(
            torch.cat([x_a[0, :5], x_b[0, :3]], dim=0),
            rope,
            torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]),
            torch.tensor([0, 0, 0, 0, 0, 5, 5, 5]),
            torch.tensor([5, 3]),
            held_win,
            ckv,
            ck,
            (p_kv, p_sc),
            SharedSelection(),
        )
        torch.testing.assert_close(out1[:5], ref_a_1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out1[5:], ref_b_1, rtol=1e-5, atol=1e-5)

        # Packed step 2: 1 decode token each
        out2, _, _, _, _ = attn.forward_packed(
            torch.cat([x_a[0, 5:6], x_b[0, 3:4]], dim=0),
            rope,
            torch.tensor([5, 3]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([6, 4]),
            held_win,
            ckv,
            ck,
            partial,
            SharedSelection(),
        )
        torch.testing.assert_close(out2[0:1], ref_a_2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out2[1:2], ref_b_2, rtol=1e-5, atol=1e-5)


def test_text_cache_clears_bound_layer_after_write_layer() -> None:
    import pytest

    cfg = DeepSeekV41Config.reduced()
    rope = DeepSeekV41Rope(cfg)
    layers = [DeepSeekV41Attention(cfg, i).eval() for i in range(3)]
    cache = DeepSeekV41TextCache(cfg, 1, 32, torch.device("cpu"), torch.float32)
    x = torch.randn(1, 3, cfg.hidden_size, dtype=torch.float32)

    with torch.no_grad():
        cache.bind_layer(0)
        layers[0](x, 0, rope, cache, SharedSelection())
        # Calling layer 1 or layer 2 without another bind_layer must fail loudly instead of reusing layer 0.
        with pytest.raises(RuntimeError, match="bind_layer"):
            layers[1](x, 0, rope, cache, SharedSelection())
        with pytest.raises(RuntimeError, match="bind_layer"):
            layers[2](x, 0, rope, cache, SharedSelection())
