"""The GLM-5.2 MTP module on CPU: its indexer-slot guard, its forward over the node's resources
and the load of the layer-78 keys."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from mstar.model.glm52.config import Glm52ModelConfig


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    """Force the CPU stub for the duration of every test in this module."""
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())
    install_cpu_rope(monkeypatch)

from mstar.engine.resources import (  # noqa: E402
    AttentionStep,
    DSAStep,
    KVStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.model.glm52._testing import build_cpu_resources, install_cpu_rope  # noqa: E402
from mstar.model.glm52.components.mtp import Glm52MTPModule  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    DSA_RESOURCE,
    KV_RESOURCE,
)
from mstar.model.utils import process_weights_after_loading  # noqa: E402


def _reduced_mtp_config() -> Glm52ModelConfig:
    """reduced() puts the MTP layer on a SHARED indexer slot; 4 trunk layers put it on a FULL
    one, as layer 78 is."""
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4
    cfg.dsa_long_context = True  # the indexer (and its checkpoint keys) exist only on the DSA path
    return cfg


def test_mtp_module_refuses_shared_position():
    with pytest.raises(ValueError, match="FULL indexer"):
        Glm52MTPModule(Glm52ModelConfig.reduced())


# --- the module forward over the bound resources ---------------------------


def _plane_step(runner, rid, span):
    steps = {KV_RESOURCE: KVStep(), ATTN_RESOURCE: AttentionStep(causal=True)}
    if DSA_RESOURCE in runner.resources:  # dsa_long_context: the layers select through it
        steps[DSA_RESOURCE] = DSAStep()
    step = SubmoduleStep(segments=[Segment(rid, "main", span)], steps=steps)
    step.set_ctx(StepContext(request_ids=(rid,), graph_walk="decode", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    return step


def test_mtp_forward_writes_the_plane_layer():
    """The module's KV goes to the extra plane at ``num_hidden_layers``, never a trunk layer's."""
    cfg = _reduced_mtp_config()
    cfg.mla_absorb = True
    cfg.mtp_num_draft_tokens = 2
    torch.manual_seed(1)
    module = Glm52MTPModule(cfg)
    # every parameter is torch.empty at construction (the loader fills them)
    for p in module.parameters():
        p.data.normal_(0, 0.05)
    process_weights_after_loading(module, torch.device("cpu"))  # w_kc / w_vc
    resources, runner = build_cpu_resources(cfg, ["r0"])
    for m in module.modules():
        if hasattr(m, "bind_resources"):
            m.bind_resources(resources)
    kv = resources[KV_RESOURCE]
    assert kv.config.num_layers == cfg.num_hidden_layers + 1
    assert module.transformer_layer.self_attn.cache_layer_idx == cfg.num_hidden_layers

    n = 3
    step = _plane_step(runner, "r0", n)
    e = torch.randn(n, cfg.hidden_size)
    h = torch.randn(n, cfg.hidden_size)
    with torch.no_grad():
        head_in, raw = module(e, h, torch.arange(n))
    runner.commit(step)
    assert head_in.shape == raw.shape == (n, cfg.hidden_size)
    assert torch.isfinite(head_in).all() and torch.isfinite(raw).all()
    # the head input is the shared_head norm of the raw stream
    with torch.no_grad():
        assert torch.equal(head_in, module.shared_head(raw))
    assert not torch.equal(head_in, raw)
    # the plane holds the n latents; every trunk layer is untouched
    pages = kv._streams["r0"]["main"].page_indices
    plane = kv.layer_view(cfg.num_hidden_layers)[pages].reshape(-1, cfg.cache_latent_dim)[:n]
    assert plane.abs().sum(-1).gt(0).all()
    for layer in range(cfg.num_hidden_layers):
        assert not kv.layer_view(layer).any()
    assert kv._streams["r0"]["main"].stored_len == n


def test_mtp_load_end_to_end_reduced_fp8():
    """The layer-78 keys ride the trunk's load stream (fp8 dequant, stacked experts, the glue)
    onto ``mtp.``, with completeness both ways."""
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.weight_loader import load_glm52_hf_weights

    torch.manual_seed(6)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg.num_hidden_layers = 4  # MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = 2
    cfg.dsa_long_context = True  # indexer keys ride the stream only on the DSA path
    model = Glm52ForCausalLM(cfg)
    assert model.mtp is not None
    assert model.mtp.transformer_layer.self_attn.indexer is not None

    state, refs = _fabricate_checkpoint(cfg, include_mtp=True)
    loaded = load_glm52_hf_weights(
        model, iter(state), cfg.n_routed_experts,
        quant_config=cfg.quantization_config, fp8_experts=True,
        dense_fp8=cfg.dense_fp8, fp8_shared_expert=cfg.fp8_shared_expert,
        num_hidden_layers=cfg.num_hidden_layers, load_mtp=True,
    )

    # completeness both ways, the mtp.* subtree included
    params = set(dict(model.named_parameters()))
    assert any(n.startswith("mtp.") for n in params)
    assert loaded == params

    # Glue: bf16 passes through unchanged; eh_proj dequantizes exactly.
    ckpt = dict(state)
    mtp_p = f"model.layers.{cfg.num_hidden_layers}"
    for param, key in (
        (model.mtp.enorm.weight, "enorm.weight"),
        (model.mtp.hnorm.weight, "hnorm.weight"),
        (model.mtp.shared_head.norm.weight, "shared_head.norm.weight"),
    ):
        assert torch.equal(param.data, ckpt[f"{mtp_p}.{key}"].to(param.dtype))
    _, _, eh_deq = refs[f"{mtp_p}.eh_proj"]
    assert torch.equal(
        model.mtp.eh_proj.weight.data,
        eh_deq.to(model.mtp.eh_proj.weight.dtype),
    )

    # The MTP MoE's routed experts land fp8-resident like the trunk's.
    moe = model.mtp.transformer_layer.mlp
    shard = cfg.moe_intermediate_size
    g8, _, _ = refs[f"{mtp_p}.mlp.experts.0.gate_proj"]
    u8, _, _ = refs[f"{mtp_p}.mlp.experts.0.up_proj"]
    got = moe.experts.gate_up_proj_fp8.data[0]
    assert torch.equal(got[:shard], g8.view(torch.uint8))
    assert torch.equal(got[shard:], u8.view(torch.uint8))

    # The MTP module's own FULL indexer got real weights too.
    idxr = model.mtp.transformer_layer.self_attn.indexer
    _, _, wk_deq = refs[f"{mtp_p}.self_attn.indexer.wk"]
    assert torch.equal(idxr.wk.weight.data, wk_deq.to(idxr.wk.weight.dtype))
