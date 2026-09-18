"""CPU tests for the DiT scaffold's static LoRA merge on plain and fused linears.

Model-independent: a toy block whose ``to_q`` / ``to_k`` / ``to_v`` live in one
``FusedColumnLinear`` next to a plain ``to_out`` linear; adapters in diffusers, Kohya and
prefixed layouts; alpha and scale; stacking two adapters through ``apply_loras``; and the
error paths (unmapped module, half a pair, rank mismatch, unknown shard).
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, ".")

from mstar.model.components.diffusion.lora import (  # noqa: E402
    LoraSpec,
    apply_loras,
    merge_lora,
    normalize_lora_state_dict,
)
from mstar.model.components.linear import FusedColumnLinear  # noqa: E402
from mstar.model.loader.base import StackedParamRule  # noqa: E402

DIM, RANK = 16, 4


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = FusedColumnLinear(DIM, {"q": DIM, "k": DIM, "v": DIM}, bias=False)
        self.out = nn.Linear(DIM, DIM, bias=False)


class ToyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock()])


# diffusers module paths -> native parameter names, as a model's checkpoint remap would do.
def remap(name: str) -> str:
    return name.replace("transformer_blocks.", "blocks.").replace("attn.to_out.0", "out").replace("attn.", "")


RULES = [
    StackedParamRule(target_suffix=".qkv", source_suffix=".to_q", shard_id="q"),
    StackedParamRule(target_suffix=".qkv", source_suffix=".to_k", shard_id="k"),
    StackedParamRule(target_suffix=".qkv", source_suffix=".to_v", shard_id="v"),
]


def lora_pair(gen: torch.Generator, out: int = DIM, inp: int = DIM, rank: int = RANK):
    a = torch.randn(rank, inp, generator=gen)
    b = torch.randn(out, rank, generator=gen)
    return a, b


def expected(weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor, factor: float) -> torch.Tensor:
    """The reference's fused weight: one fp32 accumulate, one rounding to the weight dtype."""
    return (weight.float() + (b.float() @ a.float()) * factor).to(weight.dtype)


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = ToyDiT().to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p, dtype=torch.float32))
    return m


def test_shard_slice_follows_layout_order():
    fused = FusedColumnLinear(DIM, {"q": DIM, "k": 2 * DIM, "v": DIM}, bias=False)
    assert fused.shard_slice("q") == (0, DIM)
    assert fused.shard_slice("k") == (DIM, 2 * DIM)
    assert fused.shard_slice("v") == (3 * DIM, DIM)
    with pytest.raises(KeyError):
        fused.shard_slice("o")


def test_diffusers_layout_merges_into_fused_shards_and_plain_linear(model):
    gen = torch.Generator().manual_seed(1)
    aq, bq = lora_pair(gen)
    av, bv = lora_pair(gen)
    ao, bo = lora_pair(gen)
    sd = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": aq,
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": bq,
        "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": av,
        "transformer.transformer_blocks.0.attn.to_v.lora_B.weight": bv,
        "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": ao,
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": bo,
    }
    adapter = normalize_lora_state_dict(sd)
    assert set(adapter.layers) == {
        "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_v", "transformer_blocks.0.attn.to_out.0",
    }
    assert adapter.layers["transformer_blocks.0.attn.to_q"][2] == float(RANK)  # no alpha -> alpha = rank

    block = model.blocks[0]
    before_qkv = block.qkv.weight.detach().clone()
    before_out = block.out.weight.detach().clone()
    touched = merge_lora(model, adapter, remap, RULES)
    assert sorted(touched) == ["blocks.0.out.weight", "blocks.0.qkv.weight", "blocks.0.qkv.weight"]

    rows = {}
    for shard in ("q", "k", "v"):
        offset, size = block.qkv.shard_slice(shard)
        rows[shard] = slice(offset, offset + size)
    after_qkv = block.qkv.weight
    assert after_qkv.dtype == torch.bfloat16
    torch.testing.assert_close(after_qkv[rows["q"]], expected(before_qkv[rows["q"]], aq, bq, 1.0), rtol=0, atol=0)
    torch.testing.assert_close(after_qkv[rows["v"]], expected(before_qkv[rows["v"]], av, bv, 1.0), rtol=0, atol=0)
    torch.testing.assert_close(after_qkv[rows["k"]], before_qkv[rows["k"]], rtol=0, atol=0)
    torch.testing.assert_close(block.out.weight, expected(before_out, ao, bo, 1.0), rtol=0, atol=0)


def test_kohya_names_alpha_and_scale(model):
    gen = torch.Generator().manual_seed(2)
    a, b = lora_pair(gen)
    sd = {
        "base_model.model.transformer_blocks.0.attn.to_k.lora_down.weight": a,
        "base_model.model.transformer_blocks.0.attn.to_k.lora_up.weight": b,
        "base_model.model.transformer_blocks.0.attn.to_k.alpha": torch.tensor(8.0),
        "base_model.model.transformer_blocks.0.attn.to_k.dora_scale": torch.ones(DIM),  # ignored
    }
    adapter = normalize_lora_state_dict(sd)
    assert adapter.layers["transformer_blocks.0.attn.to_k"][2] == 8.0
    block = model.blocks[0]
    before = block.qkv.weight.detach().clone()
    merge_lora(model, adapter, remap, RULES, scale=0.5)
    offset, size = block.qkv.shard_slice("k")
    factor = 0.5 * 8.0 / RANK
    torch.testing.assert_close(
        block.qkv.weight[offset:offset + size], expected(before[offset:offset + size], a, b, factor), rtol=0, atol=0,
    )
    untouched = torch.ones(3 * DIM, dtype=torch.bool)
    untouched[offset:offset + size] = False
    torch.testing.assert_close(block.qkv.weight[untouched], before[untouched], rtol=0, atol=0)


def test_convert_keys_hook_rewrites_native_layouts(model):
    gen = torch.Generator().manual_seed(3)
    a, b = lora_pair(gen)
    sd = {"blocks.0.wo.lora_A.weight": a, "blocks.0.wo.lora_B.weight": b}

    def convert(sd):
        return {k.replace("blocks.0.wo.", "transformer_blocks.0.attn.to_out.0."): v for k, v in sd.items()}

    adapter = normalize_lora_state_dict(sd, convert_keys=convert)
    assert list(adapter.layers) == ["transformer_blocks.0.attn.to_out.0"]
    before = model.blocks[0].out.weight.detach().clone()
    merge_lora(model, adapter, remap, RULES)
    torch.testing.assert_close(model.blocks[0].out.weight, expected(before, a, b, 1.0), rtol=0, atol=0)


def test_apply_loras_stacks_adapters_in_order(model, tmp_path):
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(4)
    a1, b1 = lora_pair(gen)
    a2, b2 = lora_pair(gen)
    p1, p2 = tmp_path / "one.safetensors", tmp_path / "two.safetensors"
    key = "transformer_blocks.0.attn.to_out.0.lora_{}.weight"
    save_file({key.format("A"): a1, key.format("B"): b1}, p1)
    save_file({key.format("A"): a2, key.format("B"): b2}, p2)

    before = model.blocks[0].out.weight.detach().clone()
    apply_loras(model, [LoraSpec.parse(str(p1)), LoraSpec.parse({"path": str(p2), "scale": 0.25})], remap, RULES)
    # Each adapter rounds once: the second merge starts from the first's bf16 result.
    ref = expected(expected(before, a1, b1, 1.0), a2, b2, 0.25)
    torch.testing.assert_close(model.blocks[0].out.weight, ref, rtol=0, atol=0)


def test_error_paths(model):
    gen = torch.Generator().manual_seed(5)
    a, b = lora_pair(gen)
    with pytest.raises(ValueError, match="missing lora_B"):
        normalize_lora_state_dict({"transformer_blocks.0.attn.to_q.lora_A.weight": a})
    with pytest.raises(ValueError, match="rank"):
        normalize_lora_state_dict({
            "transformer_blocks.0.attn.to_q.lora_A.weight": a,
            "transformer_blocks.0.attn.to_q.lora_B.weight": b[:, : RANK - 1],
        })
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_lora_state_dict({"transformer_blocks.0.attn.to_q.weight": a})
    adapter = normalize_lora_state_dict({
        "transformer_blocks.0.attn.to_q.lora_A.weight": a,
        "transformer_blocks.0.attn.to_q.lora_B.weight": b,
    })
    with pytest.raises(KeyError, match="not a parameter"):
        merge_lora(model, adapter, lambda name: "nowhere." + name, RULES)
    # A shard rule pointing at a plain nn.Linear is a mapping bug, not a silent skip.
    bad_rules = [StackedParamRule(target_suffix=".out", source_suffix=".to_q", shard_id="q")]
    with pytest.raises(TypeError, match="FusedColumnLinear"):
        merge_lora(model, adapter, remap, bad_rules)
    # Delta rows must match the target rows exactly.
    wide = normalize_lora_state_dict({
        "transformer_blocks.0.attn.to_out.0.lora_A.weight": a,
        "transformer_blocks.0.attn.to_out.0.lora_B.weight": torch.randn(DIM + 1, RANK, generator=gen),
    })
    with pytest.raises(ValueError, match="does not match"):
        merge_lora(model, wide, remap, RULES)
