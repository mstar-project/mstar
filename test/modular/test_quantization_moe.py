"""The other MoE blocks' routed experts in block fp8: the shared-expert block (GLM, Qwen3.5 MoE, Qwen3-Omni's
Talker) and the expert-parallel block (each rank owns whole experts). Loaded as stored or quantized at load into
the same tensors, each EP rank its own experts, and the forward within e4m3's rounding of bf16's (CPU)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from quant_helpers import EXPERTS, HIDDEN, INTER, STACKED, group, remap
from safetensors.torch import save_file
from torch import nn

from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.components.moe import ParallelSparseMoeBlock
from mstar.model.components.quant_linear import QuantizableLinears
from mstar.model.loader import iter_safetensors_shards, load_weights_into
from mstar.model.loader.base import CheckpointRules
from mstar.quantization.resolve import Action, QuantizationRoot
from mstar.quantization.scope import node_scope
from mstar.utils.fp8_block import dequantize_fp8_block_weight, quantize_fp8_block_weight

LINEARS = QuantizableLinears(
    candidates=("layers.*.shared.experts", "layers.*.shared.shared_expert.*", "layers.*.ep.experts"),
    keep_bf16=("layers.*.shared.shared_expert_gate",),
)


class Layer(nn.Module):
    def __init__(self, cg):
        super().__init__()
        self.shared = ParallelSparseMoeBlock(
            HIDDEN, EXPERTS, 2, INTER, comm_group=cg,
            shared_expert=ParallelGatedMLP(HIDDEN, INTER, cg, reduce_results=False),
        )
        self.ep = ParallelSparseMoeBlock(HIDDEN, EXPERTS, 2, INTER, comm_group=cg, expert_parallel=True)


class Root(nn.Module):
    def __init__(self, cg=None):
        super().__init__()
        self.layers = nn.ModuleList([Layer(cg or group())])


def weights(seed=0):
    g = torch.Generator().manual_seed(seed)

    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    out = {}
    for block in ("shared", "ep"):
        p = f"layers.0.{block}"
        out[f"{p}.gate.weight"] = w(EXPERTS, HIDDEN)
        for e in range(EXPERTS):
            out[f"{p}.experts.{e}.gate_proj.weight"] = w(INTER, HIDDEN)
            out[f"{p}.experts.{e}.up_proj.weight"] = w(INTER, HIDDEN)
            out[f"{p}.experts.{e}.down_proj.weight"] = w(HIDDEN, INTER)
    out["layers.0.shared.shared_expert.gate_proj.weight"] = w(INTER, HIDDEN)
    out["layers.0.shared.shared_expert.up_proj.weight"] = w(INTER, HIDDEN)
    out["layers.0.shared.shared_expert.down_proj.weight"] = w(HIDDEN, INTER)
    out["layers.0.shared.shared_expert_gate.weight"] = w(1, HIDDEN)
    return out


def write(root: Path, fp8: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, t in weights().items():
        base = name.removesuffix(".weight")
        if fp8 and ".experts." in name:
            tensors[name], tensors[f"{base}.weight_scale_inv"] = quantize_fp8_block_weight(t)
        else:
            tensors[name] = t
    save_file(tensors, str(root / "model.safetensors"))
    config = {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}} if fp8 else {}
    (root / "config.json").write_text(json.dumps(config))
    return root


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    base = tmp_path_factory.mktemp("moe")
    return {"bf16": write(base / "bf16", False), "fp8": write(base / "fp8", True)}


def load(path, quantization=None, rank=0, size=1):
    decl = QuantizationRoot(
        node="LLM", root_type=Root, linears=LINEARS,
        rules=CheckpointRules(stacked_params=tuple(STACKED), remap=remap),
        checkpoint_dir=lambda: str(path), build=Root,
    )
    model = SimpleNamespace(get_quantization_roots=lambda: [decl])
    with node_scope(model, "LLM", {} if quantization is None else {"quantization": quantization}) as scope:
        torch.set_default_dtype(torch.bfloat16)
        try:
            root = Root(group(rank, size))
        finally:
            torch.set_default_dtype(torch.float32)
        load_weights_into(root, iter_safetensors_shards(path), stacked_params=STACKED, name_remapper=remap)
        scope.finish(root)
    return root, next(iter(scope.plans.values()))[1]


def quantized(root):
    return {k: v for k, v in root.state_dict().items() if v.dtype in (torch.uint8, torch.int32)}


@pytest.mark.parametrize("size", [1, 2])
def test_stored_experts_load_into_what_quantizing_at_load_makes(checkpoints, size):
    for rank in range(size):
        direct, plan = load(checkpoints["fp8"], None, rank, size)
        assert plan.layers["layers.0.shared.experts"].action is Action.DIRECT
        assert plan.layers["layers.0.ep.experts"].action is Action.DIRECT
        fly, plan = load(checkpoints["bf16"], "fp8_block", rank, size)
        assert plan.layers["layers.0.ep.experts"].action is Action.QUANTIZE
        # the shared expert is three parallel linears, quantized in place like any other
        assert plan.layers["layers.0.shared.shared_expert.gate_up_proj"].action is Action.QUANTIZE
        # (the stored checkpoint keeps the shared expert bf16; quantizing at load takes it too)
        a = {k: v for k, v in quantized(direct).items() if ".experts." in k}
        b = {k: v for k, v in quantized(fly).items() if ".experts." in k}
        assert a.keys() == b.keys() and len(a) == 8
        for name, tensor in a.items():
            assert torch.equal(tensor, b[name]), name


def test_each_expert_parallel_rank_owns_its_whole_experts(checkpoints):
    full = quantized(load(checkpoints["fp8"])[0])
    for rank in range(2):
        mine = quantized(load(checkpoints["fp8"], None, rank, 2)[0])
        local = EXPERTS // 2
        for name in ("gate_up_proj", "gate_up_proj_scale_inv", "down_proj", "down_proj_scale_inv"):
            key = f"layers.0.ep.experts.{name}"
            assert torch.equal(mine[key], full[key][rank * local:(rank + 1) * local]), key


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_fp8_experts_track_bf16_and_skip_padding_rows(checkpoints):
    torch.manual_seed(0)
    x = torch.randn(6, HIDDEN, dtype=torch.bfloat16)
    ref, _ = load(checkpoints["bf16"])
    fp8, _ = load(checkpoints["fp8"])
    for block in ("shared", "ep"):
        r, q = getattr(ref.layers[0], block)(x), getattr(fp8.layers[0], block)(x)
        assert _rel(q, r) < 0.05, block
    valid = torch.tensor([True, True, False, True, False, True])
    out = fp8.layers[0].ep(x, token_valid=valid)
    # a padding row routes to the sentinel: nothing is added to it
    assert torch.equal(out[~valid], torch.zeros_like(out[~valid]))
    assert torch.equal(out[valid], fp8.layers[0].ep(x)[valid])


def test_the_stored_scales_are_the_checkpoints(checkpoints):
    root, _ = load(checkpoints["fp8"])
    tensors = dict(iter_safetensors_shards(checkpoints["fp8"], quant_methods=("fp8",)))  # the stored tensors, raw
    got = root.layers[0].ep.experts.down_proj_scale_inv.view(torch.float32)[3]
    assert torch.equal(got, tensors["layers.0.ep.experts.3.down_proj.weight_scale_inv"])
    w = root.layers[0].ep.experts.down_proj[3]
    assert torch.equal(dequantize_fp8_block_weight(w, got), dequantize_fp8_block_weight(
        tensors["layers.0.ep.experts.3.down_proj.weight"], got))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
                    reason="fused_experts_fp8 needs e4m3 tensor cores")
def test_the_fused_fp8_dispatch_matches_the_reference_and_skips_padding(checkpoints):
    torch.manual_seed(0)
    root, _ = load(checkpoints["fp8"])
    x = torch.randn(37, HIDDEN, dtype=torch.bfloat16)
    valid = torch.rand(37) > 0.3
    cpu = {block: getattr(root.layers[0], block)(x) for block in ("shared", "ep")}
    cpu_masked = root.layers[0].ep(x, token_valid=valid)
    root.to("cuda")
    # fused_experts_fp8 rounds the activations to e4m3 (W8A8) where the CPU reference keeps them bf16; on an H200
    # the shared block measured 0.038, the EP block 0.044 with and without padding rows
    for block, ref in cpu.items():
        assert _rel(getattr(root.layers[0], block)(x.cuda()).cpu(), ref) < 0.05, block
    masked = root.layers[0].ep(x.cuda(), token_valid=valid.cuda()).cpu()
    assert torch.equal(masked[~valid], torch.zeros_like(masked[~valid]))
    assert _rel(masked[valid], cpu_masked[valid]) < 0.05
