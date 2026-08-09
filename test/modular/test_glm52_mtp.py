"""CPU tests for the GLM-5.2 MTP components (M3, Phase D).

No engine, no GPU: pins the module's construction against the checkpoint's
layer-78 key inventory, the fusion math, and the greedy verify rule the
draft loop will build on. Engine-side draft iteration / KV rewind tests
arrive with the engine half (the M2 pattern).
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from mstar.model.glm52.components.indexer import is_full_indexer_layer
from mstar.model.glm52.config import Glm52ModelConfig

# mtp.py's import chain reaches flashinfer via decoder_layer -> attention;
# stub it before the import so construction-level tests run on machines
# without CUDA wheels (same treatment as test_glm52_indexer.py, which owns
# the stub for forward-path numerics).
if "flashinfer" not in sys.modules:
    def _cpu_rmsnorm(x, weight, eps=1e-6):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    _fi = types.ModuleType("flashinfer")
    _fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    sys.modules["flashinfer"] = _fi

from mstar.model.glm52.components.mtp import (  # noqa: E402
    Glm52MTPModule,
    mtp_greedy_verify,
    remap_mtp_key,
)

# The collapsed key inventory under model.layers.78. from the real
# checkpoint's model.safetensors.index.json (experts collapsed to one
# representative index, fp8 scale companions dropped — scale pairing is
# quantization.py's business, exercised by its own goldens).
CHECKPOINT_LAYER78_SUBKEYS = [
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "mlp.experts.0.down_proj.weight",
    "mlp.experts.0.gate_proj.weight",
    "mlp.experts.0.up_proj.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.gate.weight",
    "mlp.shared_experts.down_proj.weight",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "shared_head.norm.weight",
]


def test_mtp_layer_is_formula_full():
    # The checkpoint ships an indexer under layer 78; the IndexShare skip
    # formula must classify it FULL so Glm52DecoderLayer(78) constructs one
    # without any special-casing (78 - 3 + 1 = 76, 76 % 4 == 0).
    cfg = Glm52ModelConfig()
    assert is_full_indexer_layer(cfg, cfg.num_hidden_layers)


def _expected_module_key(sub_key: str) -> str:
    """Compose remap_mtp_key (subtree routing) with the naming conventions
    the trunk loader already implements (weight_loader.glm52_name_remapper
    + fused expert stacking): shared_experts -> shared_expert, per-expert
    gate/up/down projections -> the fused stacked parameters. This mirrors,
    not reimplements, the loader — if the loader's conventions change, the
    trunk goldens break first and this map is updated with them."""
    routed = remap_mtp_key(sub_key)
    routed = routed.replace(".shared_experts.", ".shared_expert.")
    import re

    m = re.match(r"(.*)\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", routed)
    if m:
        prefix, proj = m.groups()
        fused = "down_proj" if proj == "down_proj" else "gate_up_proj"
        return f"{prefix}.experts.{fused}"
    # Shared expert is a ParallelGatedMLP: gate/up fuse into the merged
    # column-parallel gate_up_proj (stacked weight_loader, Kimi convention).
    m = re.match(r"(.*\.shared_expert)\.(gate_proj|up_proj|down_proj)\.weight$", routed)
    if m:
        prefix, proj = m.groups()
        fused = "down_proj" if proj == "down_proj" else "gate_up_proj"
        return f"{prefix}.{fused}.weight"
    return routed


def _reduced_mtp_config() -> Glm52ModelConfig:
    """reduced() puts the MTP position (layer 2, offset=1) on a SHARED
    slot; 4 trunk layers make it land FULL (4 = offset-1 + freq), matching
    the real geometry where 78 = 2 + 19·4."""
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4
    return cfg


def test_mtp_module_refuses_shared_position():
    # reduced() itself: layer 2 with offset=1 is SHARED -> indexer-less
    # would desync from a checkpoint that ships indexer weights. Loud > silent.
    with pytest.raises(ValueError, match="FULL indexer"):
        Glm52MTPModule(Glm52ModelConfig.reduced())


def test_mtp_module_covers_checkpoint_keys():
    """Every layer-78 checkpoint key must land on a real module parameter
    under subtree routing + the trunk loader's naming conventions — the
    loader contract, pinned before the MTP loader exists. The reduced
    config has fewer experts (bf16, non-fp8-resident: fused params carry
    no _fp8 suffix); expert index 0 exists in both, which is all the
    mapping needs."""
    cfg = _reduced_mtp_config()
    module = Glm52MTPModule(cfg)
    sd_keys = set(module.state_dict().keys())
    for sub_key in CHECKPOINT_LAYER78_SUBKEYS:
        mapped = _expected_module_key(sub_key)
        assert mapped in sd_keys, f"{sub_key} -> {mapped} not in module"


def test_mtp_glue_routes_direct_trunk_routes_nested():
    assert remap_mtp_key("enorm.weight") == "enorm.weight"
    assert remap_mtp_key("shared_head.norm.weight") == "shared_head.norm.weight"
    assert (
        remap_mtp_key("self_attn.q_a_proj.weight")
        == "transformer_layer.self_attn.q_a_proj.weight"
    )
    assert (
        remap_mtp_key("input_layernorm.weight")
        == "transformer_layer.input_layernorm.weight"
    )


def test_mtp_fusion_math():
    """fuse() = eh_proj([enorm(e); hnorm(h)]) — checked against the same
    ops applied by hand, and sensitive to argument order (embedding half
    first, hidden half second — the DeepSeek-V3 convention)."""
    cfg = _reduced_mtp_config()
    module = Glm52MTPModule(cfg).eval()
    torch.manual_seed(0)
    e = torch.randn(3, cfg.hidden_size)
    h = torch.randn(3, cfg.hidden_size)
    with torch.no_grad():
        out = module.fuse(e, h)
        ref = module.eh_proj(
            torch.cat([module.enorm(e), module.hnorm(h)], dim=-1)
        )
        swapped = module.eh_proj(
            torch.cat([module.hnorm(h), module.enorm(e)], dim=-1)
        )
    assert out.shape == (3, cfg.hidden_size)
    assert torch.equal(out, ref)
    assert not torch.allclose(out, swapped)


def test_mtp_module_has_no_embedding_or_head():
    # The checkpoint ships neither under layer 78 — drafts must reuse the
    # target's embed_tokens / lm_head. A module that grew its own copies
    # would silently serve garbage (loader would skip-log them as missing).
    cfg = _reduced_mtp_config()
    names = {n for n, _ in Glm52MTPModule(cfg).named_parameters()}
    assert not any("embed" in n or "lm_head" in n for n in names)


# --- greedy verify truth table ---------------------------------------------


def _verify(draft, target):
    n, tok = mtp_greedy_verify(
        torch.tensor(draft, dtype=torch.long),
        torch.tensor(target, dtype=torch.long),
    )
    return n, int(tok)


def test_verify_all_accepted_emits_bonus():
    # target agrees with every draft; the k+1-th entry is the bonus token.
    assert _verify([5, 7, 9], [5, 7, 9, 11]) == (3, 11)


def test_verify_first_mismatch_corrects():
    assert _verify([5, 7, 9], [4, 7, 9, 11]) == (0, 4)


def test_verify_mid_mismatch_truncates():
    # draft[1] wrong: accept 1, emit the target's correction; entries past
    # the mismatch are meaningless and must not affect the result.
    assert _verify([5, 8, 9], [5, 7, 999, 999]) == (1, 7)


def test_verify_k0_is_plain_decode():
    assert _verify([], [42]) == (0, 42)


def test_verify_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mtp_greedy_verify(torch.tensor([1, 2]), torch.tensor([1, 2]))


def test_verify_emission_invariant():
    # Every step emits num_accepted + 1 tokens and they are exactly the
    # target's greedy stream: the property that makes MTP-on bit-identical
    # to MTP-off at temp 0.
    draft = [3, 1, 4, 1]
    target = [3, 1, 5, 9, 2]
    n, tok = _verify(draft, target)
    emitted = draft[:n] + [tok]
    assert emitted == target[: n + 1]


# ---------------------------------------------------------------------------
# M3 weight loader path: layer-78 keys -> the ``mtp.`` submodule.
# ---------------------------------------------------------------------------

def test_mtp_flag_default_off_no_module():
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM

    cfg = Glm52ModelConfig.reduced()
    assert cfg.mtp_num_draft_tokens == 0
    model = Glm52ForCausalLM(cfg)
    assert model.mtp is None
    assert not any(n.startswith("mtp.") for n in dict(model.named_parameters()))


def test_mtp_load_end_to_end_reduced_fp8():
    """The M3 loader contract, executed: layer-78 keys ride the single-pass
    stream — fp8 dequant, fused expert stacking, glue keys — onto the mtp
    submodule, with completeness both ways."""
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.weight_loader import load_glm52_hf_weights

    torch.manual_seed(6)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg.num_hidden_layers = 4  # MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = 2
    model = Glm52ForCausalLM(cfg)
    assert model.mtp is not None
    assert model.mtp.transformer_layer.self_attn.indexer is not None

    state, refs = _fabricate_checkpoint(cfg, include_mtp=True)
    loaded = load_glm52_hf_weights(
        model, iter(state), cfg.n_routed_experts,
        quant_config=cfg.quantization_config, fp8_experts=True,
        num_hidden_layers=cfg.num_hidden_layers, load_mtp=True,
    )

    # Completeness both ways, now including the mtp.* subtree.
    params = set(dict(model.named_parameters()))
    assert any(n.startswith("mtp.") for n in params)
    assert loaded == params

    # Glue: bf16 passthrough bit-exact; eh_proj dequantized bit-exact.
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
