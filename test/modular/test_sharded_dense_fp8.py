"""CPU contract checks for load-time quantization of TP DiT projections."""

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.dense_fp8_format import DENSE_FP8
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantColumnParallelLinear,
    QuantRowParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.lora.capable import lora_capability
from mstar.model.components.quant_linear import QuantizableLinears, apply_linear_dtype
from mstar.model.ltx2_5.components.transformer import QUANTIZABLE_LINEARS, LTX2DiT
from mstar.model.ltx2_5.config import LTX2TransformerConfig


class _Group:
    world_size = 2

    def __init__(self, rank):
        self.rank = rank
        self.other = None
        self.gathered = None

    def all_reduce(self, tensor):
        return tensor + self.other

    def all_gather(self, tensor, dim):
        return torch.cat((tensor, self.gathered), dim=dim)


def _weight(shape, seed):
    return (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.12).to(torch.bfloat16)


def test_column_gather_keeps_rank_shards_and_lora_contract(monkeypatch):
    monkeypatch.setattr(type(DENSE_FP8), "unavailable_reason", lambda self, device: None)
    weight, bias = _weight((256, 128), 1), _weight((256,), 2)
    groups = [_Group(i) for i in range(2)]
    original, quantized = [], []
    for group in groups:
        layer = ColumnParallelLinear(group, 128, 256, bias=True, gather_output=True, dtype=torch.bfloat16)
        layer.weight_loader(layer.weight, weight)
        layer.weight_loader(layer.bias, bias)
        original.append(layer)
        root = nn.Module()
        root.proj = layer
        assert apply_linear_dtype(root, "dense_fp8", QuantizableLinears(("proj",), ())) == ["proj"]
        quantized.append(root.proj)
    x = _weight((4, 128), 3)
    groups[0].gathered = quantized[1]._local(x)
    groups[1].gathered = quantized[0]._local(x)
    out = quantized[0](x)
    ref = F.linear(x, weight, bias)
    assert isinstance(quantized[0], QuantColumnParallelLinear)
    assert out.shape == ref.shape and (out.float() - ref.float()).norm() / ref.float().norm() < 0.03
    assert quantized[0].lora_target() == original[0].lora_target()
    assert quantized[0].lora_local_shape() == (128, 128)
    capable = lora_capability(quantized[0])
    assert capable.capable is not None
    quantized[0].__class__ = capable.capable
    quantized[0].lora_delta = lambda inp, y: y + 1
    assert torch.equal(quantized[0](x)[..., :128], out[..., :128] + 1)


def test_column_skip_bias_add_stays_bias_free(monkeypatch):
    monkeypatch.setattr(type(DENSE_FP8), "unavailable_reason", lambda self, device: None)
    layer = ColumnParallelLinear(
        _Group(0), 128, 256, bias=True, skip_bias_add=True, dtype=torch.bfloat16,
    )
    layer.weight.data.zero_()
    layer.bias.data.fill_(3)
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    assert torch.equal(layer(x), torch.zeros(2, 128, dtype=torch.bfloat16))
    root = nn.Module()
    root.proj = layer
    apply_linear_dtype(root, "dense_fp8", QuantizableLinears(("proj",), ()))
    assert torch.equal(root.proj(x), torch.zeros(2, 128, dtype=torch.bfloat16))


def test_row_reduce_adds_bias_once_and_keeps_lora_contract(monkeypatch):
    monkeypatch.setattr(type(DENSE_FP8), "unavailable_reason", lambda self, device: None)
    weight, bias = _weight((128, 256), 4), _weight((128,), 5)
    groups = [_Group(i) for i in range(2)]
    original, quantized = [], []
    for group in groups:
        layer = RowParallelLinear(group, 256, 128, bias=True, dtype=torch.bfloat16)
        layer.weight_loader(layer.weight, weight)
        layer._bias_loader(layer.bias, bias)
        original.append(layer)
        root = nn.Module()
        root.proj = layer
        assert apply_linear_dtype(root, "dense_fp8", QuantizableLinears(("proj",), ())) == ["proj"]
        quantized.append(root.proj)
    x = _weight((4, 256), 6)
    groups[0].other = quantized[1]._local(x[..., 128:])[1]
    out = quantized[0](x[..., :128])
    ref = F.linear(x, weight, bias)
    assert isinstance(quantized[0], QuantRowParallelLinear)
    assert out.shape == ref.shape and (out.float() - ref.float()).norm() / ref.float().norm() < 0.03
    assert quantized[0].lora_target() == original[0].lora_target()
    assert quantized[0].lora_local_shape() == (128, 128)
    capable = lora_capability(quantized[0])
    assert capable.capable is not None
    quantized[0].__class__ = capable.capable
    quantized[0].lora_delta = lambda inp, y: y + 1
    assert torch.allclose(quantized[0](x[..., :128]).float(), (out + 1).float(), atol=0.008)


def test_qkv_packed_shards_keep_order_and_target(monkeypatch):
    monkeypatch.setattr(type(DENSE_FP8), "unavailable_reason", lambda self, device: None)
    weights = {part: _weight((256, 256), i + 7) for i, part in enumerate("qkv")}
    x = _weight((3, 256), 10)
    for rank in range(2):
        group = _Group(rank)
        layer = QKVParallelLinear(group, 256, 64, 4, 4, bias=False, dtype=torch.bfloat16)
        for part, weight in weights.items():
            layer.weight_loader(layer.weight, weight, part)
        target = layer.lora_target()
        root = nn.Module()
        root.proj = layer
        apply_linear_dtype(root, "dense_fp8", QuantizableLinears(("proj",), ()))
        out = root.proj(x)
        ref = torch.cat([F.linear(x, w[rank * 128:(rank + 1) * 128]) for w in weights.values()], -1)
        assert out.shape == ref.shape and (out.float() - ref.float()).norm() / ref.float().norm() < 0.03
        assert root.proj.lora_target() == target


def test_merged_column_keeps_each_checkpoint_slice(monkeypatch):
    monkeypatch.setattr(type(DENSE_FP8), "unavailable_reason", lambda self, device: None)
    weights = [_weight((256, 128), 11), _weight((256, 128), 12)]
    x = _weight((3, 128), 13)
    for rank in range(2):
        layer = MergedColumnParallelLinear(_Group(rank), 128, [256, 256], bias=False, dtype=torch.bfloat16)
        for shard, weight in enumerate(weights):
            layer.weight_loader(layer.weight, weight, shard)
        target = layer.lora_target()
        root = nn.Module()
        root.proj = layer
        apply_linear_dtype(root, "dense_fp8", QuantizableLinears(("proj",), ()))
        out = root.proj(x)
        ref = torch.cat([F.linear(x, w[rank * 128:(rank + 1) * 128]) for w in weights], -1)
        assert out.shape == ref.shape and (out.float() - ref.float()).norm() / ref.float().norm() < 0.03
        assert root.proj.lora_target() == target


def test_ltx_dit_declares_only_block_projections(monkeypatch):
    monkeypatch.setattr(type(DENSE_FP8), "unavailable_reason", lambda self, device: None)
    cfg = replace(
        LTX2TransformerConfig(), num_layers=1, num_attention_heads=2,
        audio_num_attention_heads=2, audio_attention_head_dim=128,
        cross_attention_dim=256, audio_cross_attention_dim=256,
    )
    dit = LTX2DiT(cfg).to(dtype=torch.bfloat16)
    for weight in dit.parameters():
        nn.init.normal_(weight, std=0.01)
    changed = apply_linear_dtype(dit, "dense_fp8", QUANTIZABLE_LINEARS)
    assert changed and all(name.startswith("transformer_blocks.0.") for name in changed)
    assert "transformer_blocks.0.attn1.to_qkv" in changed
    assert "transformer_blocks.0.audio_ff.down" in changed
    assert "transformer_blocks.0.attn1.to_gate_logits" not in changed
    assert isinstance(dit.transformer_blocks[0].attn1.to_qkv, QuantColumnParallelLinear)
    assert isinstance(dit.proj_in, nn.Linear)
