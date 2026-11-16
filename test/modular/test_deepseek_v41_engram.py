"""Numerical golden from DeepSeek-V4.1-Flash's published Engram.forward."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mstar.model.components.deepseek_v41.engram import DeepSeekV41Engram, DeepSeekV41EngramEmbedding


def _golden() -> dict:
    return json.loads((Path(__file__).parent / "data/deepseek_v41_engram_golden.json").read_text())


def test_engram_matches_pinned_reference_with_masked_token() -> None:
    data = _golden()
    assert data["reference_sha256"] == "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
    model = DeepSeekV41Engram(5, 64, 2, 8, 4, wkv_reference=True).eval()
    with torch.no_grad():
        model.embed.weight.weight_loader(model.embed.weight, torch.tensor(data["table"]).to(torch.float8_e4m3fn))
        model.embed.scale.weight_loader(model.embed.scale, torch.tensor(data["table_scales"]).to(torch.float8_e8m0fnu))
        model.load_wkv_pair(
            torch.tensor(data["wkv"]).to(torch.float8_e4m3fn),
            torch.tensor(data["wkv_scales"]).to(torch.float8_e8m0fnu),
        )
        model.q_weight.copy_(torch.tensor(data["q_weight"]))
        model.k_weight.copy_(torch.tensor(data["k_weight"]))
        streams = torch.tensor(data["streams"]).to(torch.bfloat16)
        observed = model(streams, torch.tensor(data["hash_ids"]), torch.tensor(data["mask"]))
        expected = torch.tensor(data["output"]).to(torch.bfloat16)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert torch.equal(observed[:, 1], streams[:, 1])
    assert not torch.equal(observed[:, 0], streams[:, 0])
    with torch.no_grad():
        model.embed.scale.fill_(0x7F)  # E8M0 1.0, unlike the reference fixture's varied scales
        assert not torch.equal(model(streams, torch.tensor(data["hash_ids"]), torch.tensor(data["mask"])), expected)


class _RankGroup:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.world_size = 2

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def test_uneven_shards_sum_to_single_rank_and_pad_last_rank() -> None:
    table = torch.arange(5 * 32).reshape(5, 32).remainder(16).to(torch.float8_e4m3fn)
    scale = torch.ones((5, 1), dtype=torch.float8_e8m0fnu)
    indices = torch.tensor([[0, 4, 2, 3, 1]])
    whole = DeepSeekV41EngramEmbedding(5, 32)
    parts = [DeepSeekV41EngramEmbedding(5, 32, _RankGroup(rank)) for rank in range(2)]
    with torch.no_grad():
        whole.weight.weight_loader(whole.weight, table)
        whole.scale.weight_loader(whole.scale, scale)
        for module in parts:
            stop = min(module.end, module.num_embeddings)
            module.weight.weight_loader(module.weight, table[module.start:stop])
            module.scale.weight_loader(module.scale, scale[module.start:stop])
    expected = whole(indices)
    actual = sum((part(indices) for part in parts))
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(parts[1].weight[-1].float()) == 0


def test_bad_table_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        DeepSeekV41EngramEmbedding(5, 31)
    table = DeepSeekV41EngramEmbedding(5, 32)
    with pytest.raises(ValueError, match="declared layout"):
        table.weight.weight_loader(table.weight, torch.empty((4, 32), dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="float8_e8m0fnu"):
        table.scale.weight_loader(table.scale, torch.zeros((5, 1), dtype=torch.uint8))


def test_dtype_conversion_keeps_fp8_storage_and_fp32_scales() -> None:
    model = DeepSeekV41Engram(5, 64, 2, 8, 4)
    with torch.no_grad():
        model.q_weight.fill_(0.1234567)
        model.k_weight.fill_(0.7654321)
        model.wkv.weight_scale_inv.fill_(0.3333333)
    before = (model.q_weight.clone(), model.k_weight.clone(), model.wkv.weight_scale_inv.clone())
    model.to(torch.bfloat16)
    assert model.embed.weight.dtype == torch.uint8
    assert model.embed.scale.dtype == torch.uint8
    for actual, original in zip((model.q_weight, model.k_weight, model.wkv.weight_scale_inv), before, strict=True):
        assert actual.dtype == torch.float32
        assert torch.equal(actual, original)
