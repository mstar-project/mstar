"""Qwen3_5Config reads a Qwen3.8-27B ``config.json`` (the Qwen3.5 architecture),
and qwen3_8_27b is wired to the Qwen3.5 model."""
import json

from mstar.model.qwen3_5.config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5Config

# Qwen/Qwen3.8-27B's config.json, trimmed to what the dataclass reads plus the
# keys new in Qwen3.8 and a few it should ignore.
_LAYER_TYPES = ([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 16


def _qwen3_8_config(**text_overrides):
    text_config = {
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "bos_token_id": 248044,
        "eos_token_id": 248044,
        "pad_token_id": None,
        "head_dim": 256,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "layer_types": _LAYER_TYPES,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "vocab_size": 248320,
        **text_overrides,
    }
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "language_model_only": False,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "tie_word_embeddings": False,
        "text_config": text_config,
    }


def _write(tmp_path, raw):
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path


def test_from_hf_reads_qwen3_8(tmp_path):
    cfg = Qwen3_5Config.from_hf(_write(tmp_path, _qwen3_8_config()))
    assert cfg.attn_output_gate
    assert not cfg.tie_word_embeddings
    assert cfg.num_hidden_layers == 64
    assert len(cfg.full_layer_indices) == 16
    assert cfg.rotary_dim == 64
    assert cfg.mrope_section == [11, 11, 10]
    assert cfg.rope_theta == 10000000
    assert cfg.image_token_id == 248056


def test_qwen3_8_27b_is_wired_to_qwen3_5():
    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY, Qwen3_5Adapter
    from mstar.cli.main import DEFAULT_CONFIGS, _repo_root
    from mstar.model.registry import MODEL_REGISTRY, model_init_kwargs

    assert MODEL_REGISTRY["qwen3_8_27b"] == MODEL_REGISTRY["qwen3_5_27b"]
    assert model_init_kwargs("qwen3_8_27b")["model_path_hf"] == "Qwen/Qwen3.8-27B"
    assert isinstance(ADAPTER_REGISTRY["qwen3_8_27b"], Qwen3_5Adapter)
    yaml_path = _repo_root() / "configs" / DEFAULT_CONFIGS["qwen3_8_27b"]
    assert yaml_path.read_text().startswith('model: "qwen3_8_27b"\n')
