"""GLM-5.2's fp8 linears under the quantization plan, on CPU: the checkpoint's block-fp8 linears load as
stored (TP shards along whole scale blocks), kv_b and the indexer dequantize, the shared expert follows
``kernels``, the block scales survive the engine's dtype cast bit for bit, and a reduced model's forward is
bitwise the bf16 model's (host tensors dequantize exactly as the loader does)."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


@pytest.fixture(autouse=True)
def _cpu_flashinfer_without_the_real_one(monkeypatch):
    # per test: left in sys.modules, the stand-in breaks find_spec("flashinfer")
    # in every file collected after this one
    if importlib.util.find_spec("flashinfer") is None:
        monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.resources import (  # noqa: E402
    AttentionStep,
    KVStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.model.components.distributed import ColumnParallelLinear, RowParallelLinear  # noqa: E402
from mstar.model.components.distributed.mlp import ParallelGatedMLP  # noqa: E402
from mstar.model.components.fp8_formats import Fp8BlockFormat  # noqa: E402
from mstar.model.glm52._testing import build_cpu_resources, install_cpu_rope, planned_load  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE, Glm52ModelConfig  # noqa: E402
from mstar.model.utils import process_weights_after_loading  # noqa: E402


@pytest.fixture
def cpu_flashinfer(monkeypatch):
    """The CPU forward's RMSNorm; on a box with real flashinfer an earlier import would
    route host tensors into GPU kernels."""
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())
    install_cpu_rope(monkeypatch)


def _cfg(block=(16, 16), kernels="reference"):
    """The reference keeps the shared expert fp8 too (kernels: auto runs it in bf16)."""
    cfg = Glm52ModelConfig.reduced_fp8(block=block)
    cfg.mla_absorb = True
    cfg.kernels = kernels
    return cfg


def _f32(t):
    return t.view(torch.float32)


def test_restore_fp32_params_leaves_the_scale_containers():
    from mstar.model.glm52.weight_loader import restore_fp32_params

    lin = ColumnParallelLinear(CommGroup.trivial(), 64, 32, bias=False)
    lin.quant_prepare(Fp8BlockFormat.for_block((16, 16)))
    lin.weight_scale_inv.data.copy_(torch.arange(8, dtype=torch.float32).view(2, 4).view(torch.int32))
    before = lin.weight_scale_inv.data.clone()
    lin.to(torch.bfloat16)  # the submodule-load autocast
    restore_fp32_params(lin)
    assert lin.weight.dtype == torch.uint8 and torch.equal(lin.weight_scale_inv.data, before)


def _full(rows, cols, block):
    """A full checkpoint tensor pair: e4m3 weight and distinct fp32 scales per block."""
    w = (torch.randn(rows, cols) * 0.1).to(torch.float8_e4m3fn)
    s = torch.arange(1, 1 + (-(-rows // block[0])) * (-(-cols // block[1])),
                     dtype=torch.float32).view(-(-rows // block[0]), -1)
    return w, s


def test_tp_shards_take_whole_scale_blocks():
    block = (16, 16)
    fmt = Fp8BlockFormat.for_block(block)
    group = CommGroup(my_global_rank=1, my_group_rank=1, group_members=[0, 1])  # rank 1 of 2

    col = ColumnParallelLinear(group, 64, 96, bias=False)
    col.quant_prepare(fmt)
    w, s = _full(96, 64, block)
    col.weight.weight_loader(col.weight, w)
    col.weight_scale_inv.weight_loader(col.weight_scale_inv, s)
    assert torch.equal(col.weight.data, w[48:].view(torch.uint8))
    assert torch.equal(_f32(col.weight_scale_inv.data), s[3:])

    row = RowParallelLinear(group, 64, 32, bias=False)
    row.quant_prepare(fmt)
    w, s = _full(32, 64, block)
    row.weight.weight_loader(row.weight, w)
    row.weight_scale_inv.weight_loader(row.weight_scale_inv, s)
    assert torch.equal(row.weight.data, w[:, 32:].view(torch.uint8))
    assert torch.equal(_f32(row.weight_scale_inv.data), s[:, 2:])

    gu = ParallelGatedMLP(64, 64, group).gate_up_proj
    gu.quant_prepare(fmt)
    (gw, gs), (uw, us) = _full(64, 64, block), _full(64, 64, block)
    for sid, (w, s) in enumerate(((gw, gs), (uw, us))):
        gu.weight.weight_loader(gu.weight, w, sid)
        gu.weight_scale_inv.weight_loader(gu.weight_scale_inv, s, sid)
    assert torch.equal(gu.weight.data, torch.cat([gw[32:], uw[32:]]).view(torch.uint8))
    assert torch.equal(_f32(gu.weight_scale_inv.data), torch.cat([gs[2:], us[2:]]))

    # 20 rows a rank split a 16-row block: the boot check refuses it, the loader too
    bad = ColumnParallelLinear(group, 64, 40, bias=False)
    bad.quant_prepare(fmt)
    with pytest.raises(ValueError, match="scale block"):
        bad.weight_scale_inv.weight_loader(bad.weight_scale_inv, _full(40, 64, block)[1])


def _load(cfg, state, quantization=None):
    model = Glm52ForCausalLM(cfg)
    loaded = planned_load(model, state, quantization=quantization)
    return model, loaded


def test_load_keeps_fp8_bytes_and_dequantizes_kv_b():
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(2)
    cfg = _cfg(block=BLOCK)
    state, refs = _fabricate_checkpoint(cfg)
    model, loaded = _load(cfg, state)
    assert loaded == set(dict(model.named_parameters()))

    attn = model.model.layers[1].self_attn
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        w8, s, _ = refs[f"model.layers.1.self_attn.{name}"]
        assert getattr(attn, name).quant_format.block == BLOCK, name
        assert torch.equal(getattr(attn, name).weight.data, w8.view(torch.uint8)), name
        assert torch.equal(_f32(getattr(attn, name).weight_scale_inv.data), s), name
    _, _, kv_b = refs["model.layers.1.self_attn.kv_b_proj"]
    assert torch.equal(attn.kv_b_proj.weight.data, kv_b.float())

    shared = model.model.layers[1].mlp.shared_expert.gate_up_proj
    g8, gs, _ = refs["model.layers.1.mlp.shared_experts.gate_proj"]
    u8, us, _ = refs["model.layers.1.mlp.shared_experts.up_proj"]
    assert torch.equal(shared.weight.data, torch.cat([g8, u8]).view(torch.uint8))
    assert torch.equal(_f32(shared.weight_scale_inv.data), torch.cat([gs, us]))
    d8, ds, _ = refs["model.layers.0.mlp.down_proj"]
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight.data, d8.view(torch.uint8))
    assert torch.equal(_f32(model.model.layers[0].mlp.down_proj.weight_scale_inv.data), ds)

    process_weights_after_loading(model, torch.device("cpu"))
    a8, as_, _ = refs["model.layers.1.self_attn.q_a_proj"]
    k8, ks, _ = refs["model.layers.1.self_attn.kv_a_proj_with_mqa"]
    assert attn.fp8_block == BLOCK
    assert torch.equal(attn.fused_qkv_a_proj_weight, torch.cat([a8, k8]).view(torch.uint8))
    assert torch.equal(_f32(attn.fused_qkv_a_proj_scale_inv), torch.cat([as_, ks]))
    assert attn.q_a_proj.weight.numel() == 0 and attn.q_a_proj.weight_scale_inv.numel() == 0
    # kernels: reference runs the stored fp8 dequantized per call
    assert attn.o_proj.quant_format.reference


def test_load_refuses_a_linear_without_its_scales():
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(3)
    cfg = _cfg(block=BLOCK)
    state, _ = _fabricate_checkpoint(cfg)
    state = [(k, v) for k, v in state if k != "model.layers.0.self_attn.o_proj.weight_scale_inv"]
    # e4m3 with no scale is no format: refused by name before anything loads
    with pytest.raises(ValueError, match=r"layers\.0\.self_attn\.o_proj"):
        _load(cfg, state)


def test_shared_expert_stays_bf16_under_auto_kernels():
    """The decode chain (kernels: auto) runs the shared expert, faster from bf16 weights:
    the shared expert loads dequantized, everything else stays fp8."""
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(6)
    cfg = _cfg(block=BLOCK, kernels="auto")
    state, refs = _fabricate_checkpoint(cfg)
    model, loaded = _load(cfg, state)
    assert loaded == set(dict(model.named_parameters()))
    moe = model.model.layers[1].mlp
    assert moe.shared_expert.gate_up_proj.quant_format is None
    _, _, g = refs["model.layers.1.mlp.shared_experts.gate_proj"]
    inter = cfg.moe_intermediate_size
    assert torch.equal(moe.shared_expert.gate_up_proj.weight.data[:inter], g.float())
    assert model.model.layers[1].self_attn.o_proj.quant_format.name == "fp8_block"
    assert model.model.layers[0].mlp.gate_up_proj.quant_format.name == "fp8_block"
    assert moe.experts_fp8_block == BLOCK


def test_the_block_scales_survive_the_engine_dtype_cast(tmp_path):
    """The engine casts a loaded node to its dtype (bf16). The block scales sit in int32 containers and
    GLM's submodule refuses the cast, so every scale serves bit for bit as the checkpoint stores it."""
    from safetensors.torch import save_file
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    from mstar.model.glm52.glm52_model import Glm52Model
    from mstar.quantization.scope import node_scope

    torch.manual_seed(7)
    state, refs = _fabricate_checkpoint(_cfg(block=BLOCK))
    save_file({k: v.contiguous() for k, v in state}, str(tmp_path / "model.safetensors"))
    model = Glm52Model(str(tmp_path), config_variant="reduced_fp8", kernels="reference", tokenizer_mode="byte")
    with node_scope(model, "LLM", {}) as scope:
        sub = model.get_submodule("LLM", "cpu", autocast_dtype=torch.bfloat16)
        scope.finish(sub)
    sub = sub.to(device="cpu", dtype=torch.bfloat16)  # engine_manager's cast
    lm = sub.language_model
    attn = lm.model.layers[0].self_attn
    _, os_, _ = refs["model.layers.0.self_attn.o_proj"]
    assert attn.o_proj.weight_scale_inv.dtype == torch.int32
    assert torch.equal(_f32(attn.o_proj.weight_scale_inv.data), os_)
    experts = lm.model.layers[1].mlp.experts
    _, es, _ = refs["model.layers.1.mlp.experts.2.down_proj"]
    assert torch.equal(_f32(experts.down_proj_scale_inv.data)[2], es)
    assert lm.model.layers[1].mlp.gate.e_score_correction_bias.dtype == torch.float32


def _forward_cpu(model, cfg, ids):
    resources, runner = build_cpu_resources(cfg, ["r0"], page_size=8)
    for layer in model.model.layers:
        layer.self_attn.bind_resources(resources)
    n = ids.shape[0]
    step = SubmoduleStep(segments=[Segment("r0", "main", n)],
                         steps={KV_RESOURCE: KVStep(), ATTN_RESOURCE: AttentionStep(causal=True)})
    step.set_ctx(StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    with torch.no_grad():
        logits = model(ids, torch.arange(n))
    runner.commit(step)
    return logits


def test_cpu_forward_is_bitwise_the_bf16_model(cpu_flashinfer):
    """Host tensors dequantize as the loader does, so the logits are the bf16 model's bit for
    bit only if every shard, the fused q_a/kv_a scales and the SwiGLU halves line up."""
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(4)
    state, _ = _fabricate_checkpoint(_cfg(block=BLOCK))
    logits = []
    # every non-expert linear dequantized at load, then as stored (the experts fp8 in both)
    experts_fp8 = [{"match": "model.layers.*.mlp.experts", "format": "fp8_block"}]
    for quantization in ({"format": "bf16", "layers": experts_fp8}, None):
        cfg = _cfg(block=BLOCK)
        model, _ = _load(cfg, state, quantization)
        process_weights_after_loading(model, torch.device("cpu"))
        ids = torch.randint(0, cfg.vocab_size, (9,), generator=torch.Generator().manual_seed(5))
        logits.append(_forward_cpu(model.eval(), cfg, ids))
    assert torch.isfinite(logits[0]).all()
    assert torch.equal(logits[0], logits[1])
