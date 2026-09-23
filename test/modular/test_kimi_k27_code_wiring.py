import json

import pytest
import torch
from torch import nn

from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model
from mstar.model.kimi_k2_7.weight_loader import (
    build_kimi_stacked_params,
    kimi_name_remapper,
)
from mstar.model.loader.base import _apply_stacked


def test_k27_code_config_full_dims_packed_and_beta_fast():
    cfg = KimiK2Config.k27_code()

    assert cfg.moe_in_kernel_dequant is True
    assert cfg.quantization_config is None

    assert cfg.rope_scaling["beta_fast"] == 32.0
    assert cfg.rope_scaling["factor"] == 64.0
    assert cfg.rope_scaling["rope_type"] == "deepseek_yarn"

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

    base = KimiK2Config()
    assert cfg.num_hidden_layers == base.num_hidden_layers
    assert cfg.rope_scaling == base.rope_scaling  # NO beta_fast override
    assert base.moe_in_kernel_dequant is False and cfg.moe_in_kernel_dequant is True

def _route(name, stacked):
    mapped = kimi_name_remapper(name)
    if mapped is None:
        return None, None
    return _apply_stacked(mapped, stacked)


def test_remapper_language_model_prefix_and_packed_experts():
    cfg = KimiK2Config.reduced_quantized_inkernel()
    with torch.device("meta"):
        model = KimiForCausalLM(cfg)
    params = set(dict(model.named_parameters()).keys())
    stacked = build_kimi_stacked_params(cfg.n_routed_experts, packed_experts=True)

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

    shared = kimi_name_remapper(
        "language_model.model.layers.1.mlp.shared_experts.down_proj.weight"
    )
    assert shared == "model.layers.1.mlp.shared_expert.down_proj.weight"
    assert shared in params

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

    for vkey in (
        "vision_tower.encoder.blocks.0.wqkv.weight",
        "mm_projector.proj.0.weight",
    ):
        assert kimi_name_remapper(vkey) == vkey  # identity — no surgery
        target, _ = _route(vkey, stacked)
        assert target not in params  # dropped

    ws_target, _ = _route(
        "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_shape", stacked
    )
    assert ws_target not in params

    assert (
        kimi_name_remapper("model.layers.0.self_attn.q_a_proj.weight")
        == "model.layers.0.self_attn.q_a_proj.weight"
    )

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
    (tmp_dir / "config.json").write_text(json.dumps(config_dict))
    model = object.__new__(KimiK2Model)
    model.config = KimiK2Config()
    return model


def test_nested_quant_config_read(tmp_path):
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
    d = tmp_path / "flat"
    d.mkdir()
    model = _make_model_with_config_json(d, {"quantization_config": _QUANT_BLOCK})
    model._maybe_apply_checkpoint_quant_config(str(d))
    qc = model.config.quantization_config
    assert qc is not None
    assert qc.num_bits == 4
    assert qc.group_size == 32


def test_plain_bf16_config_stays_none(tmp_path):
    d = tmp_path / "bf16"
    d.mkdir()
    model = _make_model_with_config_json(d, {"text_config": {"num_hidden_layers": 61}})
    model._maybe_apply_checkpoint_quant_config(str(d))
    assert model.config.quantization_config is None


# --- KimiK2Config.from_checkpoint (A1) -------------------------------------

_FLAT_CONFIG = {
    "vocab_size": 111,
    "hidden_size": 222,
    "num_hidden_layers": 3,
    "rms_norm_eps": 2e-6,
    "n_routed_experts": 7,
    "rope_theta": 123.0,
}


def test_from_checkpoint_flat_style_overrides_fields(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_FLAT_CONFIG))
    cfg = KimiK2Config.from_checkpoint(tmp_path)
    assert cfg.vocab_size == 111
    assert cfg.hidden_size == 222
    assert cfg.num_hidden_layers == 3
    assert cfg.rms_norm_eps == 2e-6
    assert cfg.n_routed_experts == 7
    assert cfg.rope_theta == 123.0
    # Fields absent from config.json keep the dataclass default.
    assert cfg.qk_nope_head_dim == KimiK2Config().qk_nope_head_dim


def test_from_checkpoint_text_config_style(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "kimi_k25",
        "text_config": {**_FLAT_CONFIG, "quantization_config": _QUANT_BLOCK},
    }))
    cfg = KimiK2Config.from_checkpoint(tmp_path)
    assert cfg.hidden_size == 222
    assert cfg.quantization_config is not None
    assert cfg.quantization_config.num_bits == 4


def test_from_checkpoint_missing_config_json_warns_and_defaults(tmp_path, caplog):
    base = KimiK2Config.reduced()
    with caplog.at_level("WARNING"):
        cfg = KimiK2Config.from_checkpoint(tmp_path, base=base)
    assert cfg is base
    assert "no config.json" in caplog.text


def test_from_checkpoint_tie_word_embeddings_raises(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        KimiK2Config.from_checkpoint(tmp_path)


def test_from_checkpoint_non_silu_hidden_act_raises(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_act": "gelu"}))
    with pytest.raises(ValueError, match="hidden_act"):
        KimiK2Config.from_checkpoint(tmp_path)


def test_from_checkpoint_preserves_base_moe_in_kernel_dequant(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 5}))
    base = KimiK2Config.k27_code()
    cfg = KimiK2Config.from_checkpoint(tmp_path, base=base)
    assert cfg.num_hidden_layers == 5
    assert cfg.moe_in_kernel_dequant is True


def test_from_checkpoint_generation_config_eos_list_and_sampling(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({}))
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "eos_token_id": [1, 2, 3], "temperature": 0.6, "top_p": 0.9,
    }))
    cfg = KimiK2Config.from_checkpoint(tmp_path)
    assert cfg.eos_token_ids == [1, 2, 3]
    assert cfg.temperature == 0.6
    assert cfg.top_p == 0.9


def test_from_checkpoint_generation_config_eos_int(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 42}))
    cfg = KimiK2Config.from_checkpoint(tmp_path)
    assert cfg.eos_token_ids == [42]


def test_from_checkpoint_eos_token_ids_default_from_config_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 99}))
    cfg = KimiK2Config.from_checkpoint(tmp_path)
    assert cfg.eos_token_id == 99
    assert cfg.eos_token_ids == [99]


def test_from_checkpoint_config_json_eos_token_id_list_not_nested(tmp_path):
    # Some HF configs (unlike generation_config.json) put a list directly on
    # config.json's eos_token_id; it must not get double-wrapped into [[...]].
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": [7, 8, 9]}))
    cfg = KimiK2Config.from_checkpoint(tmp_path)
    assert cfg.eos_token_ids == [7, 8, 9]
    assert cfg.eos_token_id == 7


# --- __init__ checkpoint-config wiring (A1, applied before get_node_resources) --


def test_init_loads_checkpoint_config_from_local_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 5, "hidden_size": 99, "qk_rope_head_dim": 16,
        "kv_lora_rank": 32,
    }))

    model = KimiK2Model(model_path_hf=str(tmp_path), config_variant="k27_code")

    assert model.config.num_hidden_layers == 5
    assert model.config.hidden_size == 99
    assert model.config.moe_in_kernel_dequant is True  # base (k27_code) preserved

    # get_node_resources() (called before get_submodule() in engine_manager.py)
    # must see the checkpoint-derived KV cache shape — never get_submodule().
    kv_spec = model.get_node_resources()[0]
    assert kv_spec.config.num_layers == 5
    assert kv_spec.config.head_dim == 32 + 16  # kv_lora_rank + qk_rope_head_dim


def test_init_repo_id_downloads_only_config_files(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 5}))

    def fake_hf_hub_download(repo_id, filename, cache_dir=None):
        assert repo_id == "org/kimi-checkpoint"
        assert filename in ("config.json", "generation_config.json")
        if filename == "generation_config.json":
            from huggingface_hub.errors import EntryNotFoundError
            raise EntryNotFoundError("no generation_config.json")
        return str(tmp_path / filename)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download,
    )

    model = KimiK2Model(model_path_hf="org/kimi-checkpoint", config_variant="full")

    assert model.config.num_hidden_layers == 5


def test_init_failed_download_keeps_defaults_and_warns(monkeypatch, caplog):
    def failing_hf_hub_download(repo_id, filename, cache_dir=None):
        raise OSError("network unreachable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failing_hf_hub_download)

    with caplog.at_level("WARNING"):
        model = KimiK2Model(model_path_hf="org/kimi-checkpoint", config_variant="full")

    assert model.config == KimiK2Config()
    assert "Error downloading config.json" in caplog.text


def test_init_reduced_variant_does_not_touch_filesystem_or_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("reduced variant must not resolve a checkpoint config")

    monkeypatch.setattr(
        "mstar.model.kimi_k2_7.kimi_model._resolve_checkpoint_config_dir", boom,
    )

    model = KimiK2Model(model_path_hf="org/kimi-checkpoint", config_variant="reduced")
    assert model.config == KimiK2Config.reduced()


# --- _create_submodule wiring + weight completeness check (A1 + A3) --------


class _StubCausalLM(nn.Module):
    """Stands in for KimiForCausalLM: one real parameter to load, no kernels."""

    def __init__(self, config, comm_group=None):
        super().__init__()
        self.config = config
        self.p = nn.Parameter(torch.zeros(2))
        self.lm_head = nn.Identity()


def _make_checkpoint_model(tmp_path, config_dict, config_variant="full"):
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    model = object.__new__(KimiK2Model)
    model.model_path_hf = str(tmp_path)
    model.cache_dir = None
    model._config_variant = config_variant
    model.config = (
        KimiK2Config.k27_code() if config_variant == "k27_code" else KimiK2Config()
    )
    model._submodule_cache = {}
    return model


def test_create_submodule_loads_weights_for_full_variant(tmp_path, monkeypatch):
    # self.config is resolved in __init__ now (see test_init_loads_checkpoint_config_
    # from_local_dir); _create_submodule only builds the module and loads weights
    # against whatever config it's given.
    monkeypatch.setattr(
        "mstar.model.kimi_k2_7.components.causal_lm.KimiForCausalLM", _StubCausalLM,
    )
    monkeypatch.setattr(
        "mstar.model.loader.load_weights",
        lambda module, source, device="cpu": {"p"},
    )
    model = _make_checkpoint_model(tmp_path, {})

    submodule = model._create_submodule("LLM", "cpu")

    assert submodule.language_model.p.shape == (2,)


def test_create_submodule_raises_on_missing_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mstar.model.kimi_k2_7.components.causal_lm.KimiForCausalLM", _StubCausalLM,
    )
    monkeypatch.setattr(
        "mstar.model.loader.load_weights",
        lambda module, source, device="cpu": set(),
    )
    model = _make_checkpoint_model(tmp_path, {})

    with pytest.raises(RuntimeError, match="not loaded"):
        model._create_submodule("LLM", "cpu")


def test_load_kimi_hf_weights_logs_unmatched_checkpoint_keys(caplog):
    from mstar.model.kimi_k2_7.weight_loader import load_kimi_hf_weights

    # A trivial module with one real parameter: enough to exercise
    # load_kimi_hf_weights' unmatched-key bookkeeping without needing a full
    # (shard-id-requiring) Kimi checkpoint.
    module = nn.Module()
    module.p = nn.Parameter(torch.zeros(2))
    weights = [
        ("p", torch.ones(2)),
        ("vision_tower.blah.weight", torch.zeros(1)),
    ]

    with caplog.at_level("INFO"):
        loaded = load_kimi_hf_weights(module, weights, n_routed_experts=4)

    assert loaded == {"p"}
    assert torch.equal(module.p, torch.ones(2))
    assert "1 checkpoint key" in caplog.text
