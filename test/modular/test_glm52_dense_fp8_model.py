"""GLM-5.2 ``dense_fp8`` in the model, on CPU: the flag, the module layout, TP shards along
whole scale blocks, the load, and a reduced model whose forward is bitwise the bf16
model's (host tensors dequantize to bf16 exactly as the loader does)."""
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


if "flashinfer" not in sys.modules:
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        sys.modules["flashinfer"] = _cpu_flashinfer()

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.resources import (  # noqa: E402
    AttentionStep,
    KVStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.model.components.distributed import ColumnParallelLinear  # noqa: E402
from mstar.model.glm52._testing import build_cpu_resources  # noqa: E402
from mstar.model.glm52.components.attention import Glm52MLAAttention  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.components.fp8_linear import (  # noqa: E402
    COLUMN,
    ROW,
    Fp8Linear,
    Fp8ParallelGatedMLP,
)
from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE, Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.quantization import process_weights_after_loading  # noqa: E402


@pytest.fixture
def cpu_flashinfer(monkeypatch):
    """The CPU forward's RMSNorm; on a box with real flashinfer an earlier import would
    route host tensors into GPU kernels."""
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


def _cfg(dense_fp8=True, block=(16, 16)):
    cfg = Glm52ModelConfig.reduced_fp8(block=block)
    cfg.mla_absorb = True
    cfg.dense_fp8 = dense_fp8
    return cfg


def test_flag_defaults_off_and_reaches_the_config():
    from mstar.model.glm52.glm52_model import Glm52Model

    assert Glm52ModelConfig().dense_fp8 is False
    assert Glm52Model("unused", config_variant="reduced_fp8").config.dense_fp8 is False
    model = Glm52Model("unused", config_variant="reduced_fp8", dense_fp8=True)
    assert model.config.dense_fp8 is True


def test_flag_needs_the_fp8_checkpoint():
    cfg = Glm52ModelConfig.reduced()
    cfg.dense_fp8 = True
    with pytest.raises(ValueError, match="quantization_config"):
        Glm52MLAAttention(cfg)


def test_flag_builds_fp8_linears_except_kv_b():
    model = Glm52ForCausalLM(_cfg())
    attn = model.model.layers[0].self_attn
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        lin = getattr(attn, name)
        assert isinstance(lin, Fp8Linear), name
        assert lin.weight.dtype == torch.uint8 and lin.weight_scale_inv.dtype == torch.float32
    assert isinstance(attn.kv_b_proj, ColumnParallelLinear)  # folded into bf16 w_kc / w_vc
    assert tuple(attn.kv_a_proj_with_mqa.weight_scale_inv.shape) == (3, 8)  # 40 rows: ragged
    assert isinstance(model.model.layers[0].mlp, Fp8ParallelGatedMLP)  # dense layer
    assert isinstance(model.model.layers[1].mlp.shared_expert, Fp8ParallelGatedMLP)
    assert model.model.layers[1].mlp.shared_expert.down_proj.reduce_results

    off = Glm52ForCausalLM(_cfg(dense_fp8=False))
    assert isinstance(off.model.layers[0].self_attn.q_a_proj, torch.nn.Linear)
    assert not any(isinstance(m, Fp8Linear) for m in off.modules())


def test_bf16_autocast_leaves_the_fp8_layout_to_restore():
    from mstar.model.glm52.weight_loader import restore_fp32_params

    lin = Fp8Linear(64, 32, (16, 16))
    lin.to(torch.bfloat16)  # the submodule-load autocast
    assert lin.weight.dtype == torch.uint8
    restore_fp32_params(lin)
    assert lin.weight_scale_inv.dtype == torch.float32


def _full(rows, cols, block):
    """A full checkpoint tensor pair: e4m3 weight and distinct fp32 scales per block."""
    w = (torch.randn(rows, cols) * 0.1).to(torch.float8_e4m3fn)
    s = torch.arange(1, 1 + (-(-rows // block[0])) * (-(-cols // block[1])),
                     dtype=torch.float32).view(-(-rows // block[0]), -1)
    return w, s


def test_tp_shards_take_whole_scale_blocks():
    block = (16, 16)
    group = CommGroup(my_global_rank=1, my_group_rank=1, group_members=[0, 1])  # rank 1 of 2

    col = Fp8Linear(64, 96, block, group, shard=COLUMN)
    w, s = _full(96, 64, block)
    col.weight.weight_loader(col.weight, w)
    col.weight_scale_inv.weight_loader(col.weight_scale_inv, s)
    assert torch.equal(col.weight.data, w[48:].view(torch.uint8))
    assert torch.equal(col.weight_scale_inv.data, s[3:])

    row = Fp8Linear(64, 32, block, group, shard=ROW, reduce_results=True)
    w, s = _full(32, 64, block)
    row.weight.weight_loader(row.weight, w)
    row.weight_scale_inv.weight_loader(row.weight_scale_inv, s)
    assert torch.equal(row.weight.data, w[:, 32:].view(torch.uint8))
    assert torch.equal(row.weight_scale_inv.data, s[:, 2:])

    mlp = Fp8ParallelGatedMLP(64, 64, block, group)
    gu = mlp.gate_up_proj
    (gw, gs), (uw, us) = _full(64, 64, block), _full(64, 64, block)
    for sid, (w, s) in enumerate(((gw, gs), (uw, us))):
        gu.weight.weight_loader(gu.weight, w, sid)
        gu.weight_scale_inv.weight_loader(gu.weight_scale_inv, s, sid)
    assert torch.equal(gu.weight.data, torch.cat([gw[32:], uw[32:]]).view(torch.uint8))
    assert torch.equal(gu.weight_scale_inv.data, torch.cat([gs[2:], us[2:]]))

    with pytest.raises(AssertionError, match="scale block"):
        Fp8Linear(64, 40, block, group, shard=COLUMN)  # 20 rows per rank
    with pytest.raises(TypeError, match="fp8"):
        col.weight.weight_loader(col.weight, torch.zeros(96, 64, dtype=torch.bfloat16))


def _load(cfg, state):
    model = Glm52ForCausalLM(cfg)
    loaded = model.load_weights(iter(state))
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
        assert torch.equal(getattr(attn, name).weight.data, w8.view(torch.uint8)), name
        assert torch.equal(getattr(attn, name).weight_scale_inv.data, s), name
    _, _, kv_b = refs["model.layers.1.self_attn.kv_b_proj"]
    assert torch.equal(attn.kv_b_proj.weight.data, kv_b.float())

    shared = model.model.layers[1].mlp.shared_expert.gate_up_proj
    g8, gs, _ = refs["model.layers.1.mlp.shared_experts.gate_proj"]
    u8, us, _ = refs["model.layers.1.mlp.shared_experts.up_proj"]
    assert torch.equal(shared.weight.data, torch.cat([g8, u8]).view(torch.uint8))
    assert torch.equal(shared.weight_scale_inv.data, torch.cat([gs, us]))
    d8, ds, _ = refs["model.layers.0.mlp.down_proj"]
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight.data, d8.view(torch.uint8))
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight_scale_inv.data, ds)

    process_weights_after_loading(model, torch.device("cpu"))
    a8, as_, _ = refs["model.layers.1.self_attn.q_a_proj"]
    k8, ks, _ = refs["model.layers.1.self_attn.kv_a_proj_with_mqa"]
    assert torch.equal(attn.fused_qkv_a_proj_weight, torch.cat([a8, k8]).view(torch.uint8))
    assert torch.equal(attn.fused_qkv_a_proj_scale_inv, torch.cat([as_, ks]))
    assert attn.q_a_proj.weight.numel() == 0 and attn.q_a_proj.weight_scale_inv.numel() == 0


def test_load_refuses_a_linear_without_its_scales():
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(3)
    cfg = _cfg(block=BLOCK)
    state, _ = _fabricate_checkpoint(cfg)
    state = [(k, v) for k, v in state if k != "model.layers.0.self_attn.o_proj.weight_scale_inv"]
    with pytest.raises(RuntimeError, match="fp8 linear tensors received no checkpoint"):
        _load(cfg, state)


def test_shared_expert_stays_bf16_under_moe_decode():
    """moe_decode runs the shared expert inside its chain, faster from bf16 weights: with
    both flags the shared expert loads dequantized, everything else stays fp8."""
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(6)
    cfg = _cfg(block=BLOCK)
    cfg.moe_decode_kernel = True
    state, refs = _fabricate_checkpoint(cfg)
    model, loaded = _load(cfg, state)
    assert loaded == set(dict(model.named_parameters()))
    moe = model.model.layers[1].mlp
    assert not isinstance(moe.shared_expert, Fp8ParallelGatedMLP)
    _, _, g = refs["model.layers.1.mlp.shared_experts.gate_proj"]
    inter = cfg.moe_intermediate_size
    assert torch.equal(moe.shared_expert.gate_up_proj.weight.data[:inter], g.float())
    assert isinstance(model.model.layers[1].self_attn.o_proj, Fp8Linear)
    assert isinstance(model.model.layers[0].mlp, Fp8ParallelGatedMLP)


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
    """On host tensors the fp8 linears dequantize to bf16 exactly as the loader does for
    the flag-off model, so every shard, the fused q_a/kv_a scales and the SwiGLU halves
    line up only if the logits agree bit for bit."""
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(4)
    state, _ = _fabricate_checkpoint(_cfg(block=BLOCK))
    logits = []
    for flag in (False, True):
        cfg = _cfg(dense_fp8=flag, block=BLOCK)
        model, _ = _load(cfg, state)
        process_weights_after_loading(model, torch.device("cpu"))
        ids = torch.randint(0, cfg.vocab_size, (9,), generator=torch.Generator().manual_seed(5))
        logits.append(_forward_cpu(model.eval(), cfg, ids))
    assert torch.isfinite(logits[0]).all()
    assert torch.equal(logits[0], logits[1])
