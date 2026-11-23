"""CPU contract checks for load-time quantization of TP DiT projections."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.dense_fp8_format import DENSE_FP8, DenseFp8Format
from mstar.model.components.diffusion.denoise_loop import StepBucket
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantColumnParallelLinear,
    QuantRowParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.lora.capable import lora_capability
from mstar.model.components.quant_linear import (
    QuantizableLinears,
    QuantLinear,
    StepDtypes,
    apply_linear_dtype,
    select_linears,
)
from mstar.model.ltx2_5.components.transformer import QUANTIZABLE_LINEARS, LTX2DiT
from mstar.model.ltx2_5.config import DENOISE_LOOP, LTX2TransformerConfig, LTX25Config
from mstar.model.ltx2_5.ltx2_5_model import GENERATE_AV_WALK, LTX25Model
from mstar.utils.fp8_block import dequantize_fp8_block_weight


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


def test_sharded_dense_fp8_tp2_vs_tp1_parity(monkeypatch):
    monkeypatch.setattr(type(DENSE_FP8), "unavailable_reason", lambda self, device: None)
    x = _weight((4, 256), 21)

    # Column-parallel TP=2 vs TP=1 QuantLinear (exact bit-for-bit parity, incl. skip_bias_add=True).
    w_col, b_col = _weight((256, 256), 22), _weight((256,), 23)
    for skip_bias in (False, True):
        tp1 = QuantLinear(w_col, None if skip_bias else b_col, DENSE_FP8)
        groups = [_Group(i) for i in range(2)]
        shards = []
        for group in groups:
            layer = ColumnParallelLinear(
                group, 256, 256, bias=True, gather_output=True, skip_bias_add=skip_bias, dtype=torch.bfloat16,
            )
            layer.weight_loader(layer.weight, w_col)
            layer.weight_loader(layer.bias, b_col)
            root = nn.Module()
            root.proj = layer
            apply_linear_dtype(root, "dense_fp8", QuantizableLinears(("proj",), ()))
            shards.append(root.proj)
        groups[0].gathered = shards[1]._local(x)
        assert torch.equal(shards[0](x), tp1(x))

    # Row-parallel TP=2 matches exact split-reduction reference bit-for-bit and TP=1 within bf16 rounding.
    w_row, b_row = _weight((128, 256), 24), _weight((128,), 25)
    tp1_row = QuantLinear(w_row, b_row, DENSE_FP8)
    groups = [_Group(i) for i in range(2)]
    row_shards = []
    for group in groups:
        layer = RowParallelLinear(group, 256, 128, bias=True, input_is_parallel=True, dtype=torch.bfloat16)
        layer.weight_loader(layer.weight, w_row)
        layer._bias_loader(layer.bias, b_row)
        root = nn.Module()
        root.proj = layer
        apply_linear_dtype(root, "dense_fp8", QuantizableLinears(("proj",), ()))
        row_shards.append(root.proj)
    groups[0].other = row_shards[1]._local(x[..., 128:])[1]
    out_tp2 = row_shards[0](x[..., :128])
    deq = dequantize_fp8_block_weight(tp1_row.weight, tp1_row.weight_scale_inv, (128, 128)).to(torch.bfloat16)
    expected_split = (F.linear(x[..., :128], deq[:, :128]) + b_row) + F.linear(x[..., 128:], deq[:, 128:])
    assert torch.equal(out_tp2, expected_split)
    torch.testing.assert_close(out_tp2, tp1_row(x), rtol=2e-2, atol=2e-2)


def test_real_ltx2_config_tp_shapes_and_tp32_refusal(monkeypatch):
    monkeypatch.setattr(DenseFp8Format, "unavailable_reason", lambda self, device: None)
    real_cfg = replace(LTX2TransformerConfig(), num_layers=1)

    for tp in (1, 2, 4, 8, 16):
        group = CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(tp)))
        with torch.device("meta"):
            dit = LTX2DiT(real_cfg, comm_group=group).to(dtype=torch.bfloat16)
        selection = select_linears(dit, "dense_fp8", QUANTIZABLE_LINEARS, what="LTX-2.5 DiT")
        assert len(selection.names) == 20

    group_32 = CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(32)))
    with torch.device("meta"):
        dit_32 = LTX2DiT(real_cfg, comm_group=group_32).to(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiples of 128"):
        apply_linear_dtype(dit_32, "dense_fp8", QUANTIZABLE_LINEARS, what="LTX-2.5 DiT")
    assert not any(isinstance(m, (QuantColumnParallelLinear, QuantRowParallelLinear)) for m in dit_32.modules())


def test_ltx25_model_linear_dtype_keep_and_steps(monkeypatch):
    from mstar.model.ltx2_5 import weight_loader

    for valid in ("bf16", "dense_fp8", "mxfp8", "nvfp4"):
        assert LTX25Model(skip_weight_loading=True, linear_dtype=valid).linear_dtype == valid
    for bad in ("fp8", "mxfp4", None):
        with pytest.raises(ValueError, match="linear_dtype must be"):
            LTX25Model(skip_weight_loading=True, linear_dtype=bad)
    with pytest.raises(ValueError, match="linear_keep_bf16"):
        LTX25Model(skip_weight_loading=True, linear_dtype="dense_fp8", linear_keep_bf16="transformer_blocks.0.*.*")
    with pytest.raises(ValueError, match="linear_dtype_steps"):
        LTX25Model(skip_weight_loading=True, linear_dtype="dense_fp8", linear_dtype_steps=-1)

    tiny_cfg = replace(
        LTX2TransformerConfig(), num_layers=1, num_attention_heads=2,
        audio_num_attention_heads=2, audio_attention_head_dim=128,
        cross_attention_dim=256, audio_cross_attention_dim=256,
    )
    tp_group = CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0])

    def fake_build(config, snapshot, device, comm_group=None):
        torch.manual_seed(0)
        dit = LTX2DiT(config.transformer, comm_group=comm_group).to(torch.bfloat16)
        with torch.no_grad():
            for p in dit.parameters():
                p.normal_(std=0.02)
        return dit

    monkeypatch.setattr(weight_loader, "build_transformer", fake_build)
    reason = None
    monkeypatch.setattr(DenseFp8Format, "unavailable_reason", lambda self, device: reason)

    def build_sub(**kwargs):
        model = LTX25Model(compile=False, accelerator_graph=False, **kwargs)
        model.set_config(LTX25Config(transformer=tiny_cfg))
        model._snapshot = Path("/nonexistent")
        return model._create_submodule("dit", torch.device("cpu"), tp_group=tp_group)

    sub = build_sub(
        linear_dtype="dense_fp8",
        linear_keep_bf16=["transformer_blocks.0.attn1.*"],
        linear_dtype_steps=1,
    )
    assert sub.step_dtypes == StepDtypes(("bf16", "dense_fp8"))
    assert sub.split_batches_by_capture_key(GENERATE_AV_WALK)
    assert isinstance(sub.transformer.transformer_blocks[0].attn1.to_qkv, ColumnParallelLinear)
    assert not isinstance(sub.transformer.transformer_blocks[0].attn1.to_qkv, QuantColumnParallelLinear)
    assert isinstance(sub.transformer.transformer_blocks[0].audio_ff.down, QuantRowParallelLinear)
    assert not any(
        isinstance(m, (QuantColumnParallelLinear, QuantRowParallelLinear))
        for m in sub.bf16_transformer.modules()
    )

    # Dispatch check: step 0 selects bf16_transformer, step 1 selects quantized transformer.
    seen = []
    orig_bf16 = sub.bf16_transformer.forward
    orig_fp8 = sub.transformer.forward
    sub.bf16_transformer.forward = lambda *a, **k: seen.append("bf16") or orig_bf16(*a, **k)
    sub.transformer.forward = lambda *a, **k: seen.append("dense_fp8") or orig_fp8(*a, **k)
    sub.get_device = lambda: torch.device("cpu")

    for k in (0, 1):
        info = SimpleNamespace(
            request_id="r", rid_handle="r", graph_walk=GENERATE_AV_WALK, random_seed=0,
            step_metadata={"height": 64, "width": 64, "num_frames": 9, "fps": 24.0},
            dynamic_loop_iter_counts={DENOISE_LOOP: k},
        )
        prepared = sub.prepare_inputs(GENERATE_AV_WALK, info, {
            "latents": [torch.randn(8, 128)],
            "audio_latents": [torch.randn(9, 128)],
            "text_video": [torch.randn(4, 256, dtype=torch.bfloat16)],
            "text_audio": [torch.randn(4, 256, dtype=torch.bfloat16)],
        })
        assert isinstance(prepared.resource_step_info, StepBucket)
        assert prepared.resource_step_info.dtype == ("bf16" if k == 0 else "dense_fp8")
        engine_inputs = SimpleNamespace(request_ids=["r"], per_request_info={"r": info})
        stacked = sub.preprocess(GENERATE_AV_WALK, engine_inputs, [prepared])
        sub.forward(GENERATE_AV_WALK, engine_inputs=engine_inputs, **stacked)
    assert seen == ["bf16", "dense_fp8"]

    # No twin when bf16 or unavailable.
    assert build_sub(linear_dtype="bf16", linear_dtype_steps=1).bf16_transformer is None
    reason = "no SM90"
    assert build_sub(linear_dtype="dense_fp8", linear_dtype_steps=1).bf16_transformer is None
