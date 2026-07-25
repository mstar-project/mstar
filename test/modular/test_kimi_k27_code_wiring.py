"""CPU wiring tests for the REAL ``moonshotai/Kimi-K2.7-Code`` text-only serve.

No weights, no GPU — the golden gate for the K2.7-Code serve plumbing while the
595 GB checkpoint is still downloading. Three concerns:

  1. :meth:`KimiK2Config.k27_code` builds the full-size 1T text config with packed
     experts armed and — crucially — keeps the default ``beta_fast=32.0`` (the
     K2.7-Code ``text_config`` value). Guards against clobbering the YaRN field.
  2. :func:`kimi_name_remapper` strips the multimodal ``language_model.`` prefix so
     the DeepSeek-V3 text keys land on ``KimiForCausalLM``'s params, routes the
     packed routed-expert sub-keys through the packed-expert stacked rules, and drops the
     vision (``vision_tower.*`` / ``mm_projector.*``) + ``weight_shape`` keys. A bare
     ``model.*`` key (no prefix) is left unchanged.
  3. ``_maybe_apply_checkpoint_quant_config`` reads the ``quantization_config``
     NESTED under ``text_config`` (the multimodal wrapper leaves the top-level
     null), while still parsing a flat top-level block and staying ``None`` for a
     plain-bf16 checkpoint.

Run:  pytest test/modular/test_kimi_k27_code_wiring.py -v
"""
import json

import torch

from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model
from mstar.model.kimi_k2_7.weight_loader import (
    build_kimi_stacked_params,
    kimi_name_remapper,
)
from mstar.model.loader.base import _apply_stacked

# --------------------------------------------------------------------------
# 1. Config: k27_code() == full 1T dims + packed experts, default beta_fast=32.0.
# --------------------------------------------------------------------------

def test_k27_code_config_full_dims_packed_and_beta_fast():
    cfg = KimiK2Config.k27_code()

    # Packed experts armed, quant config auto-read from the checkpoint (still None here).
    assert cfg.moe_in_kernel_dequant is True
    assert cfg.quantization_config is None

    # K2.7-Code keeps beta_fast=32.0 (guard against clobbering the YaRN field).
    assert cfg.rope_scaling["beta_fast"] == 32.0
    assert cfg.rope_scaling["factor"] == 64.0
    assert cfg.rope_scaling["rope_type"] == "deepseek_yarn"

    # Full 1T text dims, matching the real Kimi-K2.7-Code text_config.
    assert cfg.num_hidden_layers == 61
    assert cfg.n_routed_experts == 384
    assert cfg.hidden_size == 7168
    assert cfg.q_lora_rank == 1536
    assert cfg.kv_lora_rank == 512
    assert cfg.moe_intermediate_size == 2048
    assert cfg.routed_scaling_factor == 2.827
    assert cfg.qk_nope_head_dim == 128
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 128

    # It really is the full-size default plus exactly the one flag (no dim drift).
    base = KimiK2Config()
    assert cfg.num_hidden_layers == base.num_hidden_layers
    assert cfg.rope_scaling == base.rope_scaling  # NO beta_fast override
    assert base.moe_in_kernel_dequant is False and cfg.moe_in_kernel_dequant is True


# --------------------------------------------------------------------------
# 2. Remapper: language_model.* strip + packed-expert routing + drops.
# --------------------------------------------------------------------------

def _route(name, stacked):
    """Mirror the loader: name_remapper then stacked-shard routing."""
    mapped = kimi_name_remapper(name)
    if mapped is None:
        return None, None
    return _apply_stacked(mapped, stacked)


def test_remapper_language_model_prefix_and_packed_experts():
    cfg = KimiK2Config.reduced_quantized_inkernel()  # in-kernel dequant => packed expert params
    with torch.device("meta"):
        model = KimiForCausalLM(cfg)
    params = set(dict(model.named_parameters()).keys())
    stacked = build_kimi_stacked_params(cfg.n_routed_experts, packed_experts=True)

    # -- plain text keys: strip language_model., land on a real param ------------
    assert kimi_name_remapper(
        "language_model.model.layers.0.self_attn.q_a_proj.weight"
    ) == "model.layers.0.self_attn.q_a_proj.weight"
    assert (
        kimi_name_remapper("language_model.model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )
    assert kimi_name_remapper("language_model.lm_head.weight") == "lm_head.weight"
    for landed in (
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ):
        assert landed in params

    # -- shared expert: plural -> singular ---------------------------------------
    shared = kimi_name_remapper(
        "language_model.model.layers.1.mlp.shared_experts.down_proj.weight"
    )
    assert shared == "model.layers.1.mlp.shared_expert.down_proj.weight"
    assert shared in params

    # -- packed routed expert: remap + stacked -> the FOUR packed params ---------
    gate_p, gate_sid = _route(
        "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_packed", stacked
    )
    assert gate_p == "model.layers.1.mlp.experts.gate_up_proj_packed"
    assert gate_sid == "gate:0"
    assert gate_p in params

    scale_p, scale_sid = _route(
        "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale", stacked
    )
    assert scale_p == "model.layers.1.mlp.experts.gate_up_proj_scale"
    assert scale_sid == "gate:0"
    assert scale_p in params

    down_p, down_sid = _route(
        "language_model.model.layers.1.mlp.experts.0.down_proj.weight_packed", stacked
    )
    assert down_p == "model.layers.1.mlp.experts.down_proj_packed"
    assert down_sid == "down:0"
    assert down_p in params

    # -- vision drop: identity remap, NOT a model param (base loader skips it) ----
    for vkey in (
        "vision_tower.encoder.blocks.0.wqkv.weight",
        "mm_projector.proj.0.weight",
    ):
        assert kimi_name_remapper(vkey) == vkey  # identity — no surgery
        target, _ = _route(vkey, stacked)
        assert target not in params  # dropped

    # -- weight_shape drop: routes to no real param ------------------------------
    ws_target, _ = _route(
        "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_shape", stacked
    )
    assert ws_target not in params

    # -- a flat model.* key (no language_model. prefix) is untouched -------------
    assert (
        kimi_name_remapper("model.layers.0.self_attn.q_a_proj.weight")
        == "model.layers.0.self_attn.q_a_proj.weight"
    )


# --------------------------------------------------------------------------
# 3. Nested-quant reader: text_config.quantization_config + flat + bf16.
# --------------------------------------------------------------------------

_QUANT_BLOCK = {
    "format": "pack-quantized",
    "quant_method": "compressed-tensors",
    "ignore": ["lm_head", "re:.*self_attn.*", "re:.*shared_experts.*"],
    "config_groups": {
        "group_0": {
            "weights": {
                "num_bits": 4,
                "group_size": 32,
                "symmetric": True,
                "strategy": "group",
                "type": "int",
            },
            "targets": ["Linear"],
        }
    },
}


def _make_model_with_config_json(tmp_dir, config_dict):
    """A KimiK2Model with a bf16 (quant=None) config and a written config.json."""
    (tmp_dir / "config.json").write_text(json.dumps(config_dict))
    model = object.__new__(KimiK2Model)
    model.config = KimiK2Config()  # quantization_config is None by default
    return model


def test_nested_quant_config_read(tmp_path):
    # Nested under text_config, top-level absent — the real K2.7-Code layout.
    d = tmp_path / "nested"
    d.mkdir()
    model = _make_model_with_config_json(
        d, {"text_config": {"num_hidden_layers": 61, "quantization_config": _QUANT_BLOCK}}
    )
    model._maybe_apply_checkpoint_quant_config(str(d))
    qc = model.config.quantization_config
    assert qc is not None
    assert qc.num_bits == 4
    assert qc.group_size == 32
    assert qc.symmetric is True
    assert qc.quant_format == "pack-quantized"


def test_flat_quant_config_read_backward_compat(tmp_path):
    # A flat top-level quantization_config block must still parse.
    d = tmp_path / "flat"
    d.mkdir()
    model = _make_model_with_config_json(d, {"quantization_config": _QUANT_BLOCK})
    model._maybe_apply_checkpoint_quant_config(str(d))
    qc = model.config.quantization_config
    assert qc is not None
    assert qc.num_bits == 4
    assert qc.group_size == 32


def test_plain_bf16_config_stays_none(tmp_path):
    # No quant block anywhere (nested or flat) — stays bf16 (None).
    d = tmp_path / "bf16"
    d.mkdir()
    model = _make_model_with_config_json(d, {"text_config": {"num_hidden_layers": 61}})
    model._maybe_apply_checkpoint_quant_config(str(d))
    assert model.config.quantization_config is None
