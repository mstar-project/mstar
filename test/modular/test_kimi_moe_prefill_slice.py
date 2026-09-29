import torch

from mstar.model.kimi_k2_7.components.language_model import build_moe_block
from mstar.model.kimi_k2_7.config import KimiK2Config


def _random_block(seed: int):
    torch.manual_seed(seed)
    cfg = KimiK2Config.reduced()
    block = build_moe_block(cfg)
    for p in block.parameters():
        p.data.normal_(std=0.02)
    return cfg, block


def test_moe_prefill_slice_matches_unsliced():
    cfg, block = _random_block(0)
    slice_size = 16
    T = 3 * slice_size + 5
    h = torch.randn(T, cfg.hidden_size) * 0.1

    block.moe_prefill_slice = slice_size
    sliced = block(h)

    block.moe_prefill_slice = T + 1
    unsliced = block(h)

    torch.testing.assert_close(sliced, unsliced, rtol=1e-5, atol=1e-5)


def test_moe_small_prefill_skips_slicing():
    cfg, block = _random_block(1)
    block.moe_prefill_slice = 16
    h = torch.randn(5, cfg.hidden_size) * 0.1

    got = block(h)

    flat = h.view(-1, cfg.hidden_size)
    topk_weights, topk_ids = block.gate(flat)
    expected = (
        block._route(flat, topk_weights, topk_ids) + block.shared_expert(flat)
    ).view(h.shape)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
