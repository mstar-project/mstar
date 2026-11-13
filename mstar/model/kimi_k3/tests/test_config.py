"""Kimi K3 / K3.1 config: the release's config.json (fixtures, from the HF API at the pinned
revisions) parses to the geometry the port declares, and what it does not port is refused."""
import copy
import json
from pathlib import Path

import pytest

from mstar.model.kimi_k3.config import KimiK3ModelConfig

FIXTURES = Path(__file__).parent / "fixtures"


def _hf(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_k31_config_json_is_the_declared_geometry():
    cfg = KimiK3ModelConfig.from_hf_config(_hf("k31_config.json"))
    assert cfg == KimiK3ModelConfig.k31()
    # 1-based lists in config.json: 0-based MLA at 3, 7, ..., 91 and the last layer 92 too
    assert cfg.full_attn_layers[:2] == (3, 7) and cfg.full_attn_layers[-2:] == (91, 92)
    assert len(cfg.full_attn_layers) == 24 and len(cfg.kda_layers) == 69
    assert not cfg.is_full_attention_layer(0) and not cfg.is_moe_layer(0) and cfg.is_moe_layer(1)
    assert cfg.dsa and cfg.index_topk == 2048 and cfg.max_seq_len == 2048
    assert cfg.cache_latent_dim == 576 and cfg.qk_head_dim == 192
    assert cfg.experts_checkpoint_format == "mxfp4"
    assert (cfg.situ_beta, cfg.situ_linear_beta) == (4.0, 25.0)
    # config.json's eos (<|end_of_msg|>) and the tokenizer's [EOS] / [EOT]
    assert cfg.eos_token_ids == (163586, 163585, 163593)
    assert cfg.num_attn_res_blocks == 8


def test_k3_is_k31_without_the_indexer():
    k3 = KimiK3ModelConfig.from_hf_config(_hf("k3_config.json"))
    assert k3 == KimiK3ModelConfig.k3()
    assert not k3.dsa and k3.max_seq_len == 32768
    k31 = KimiK3ModelConfig.k31()
    same = {f for f in k3.__dataclass_fields__ if getattr(k3, f) == getattr(k31, f)}
    assert set(k3.__dataclass_fields__) - same == {"dsa", "max_seq_len"}


def test_dsa_holds_the_context_to_index_topk_unless_long_context():
    hf = _hf("k31_config.json")
    assert KimiK3ModelConfig.from_hf_config(hf, max_seq_len=8192).max_seq_len == 2048
    long = KimiK3ModelConfig.from_hf_config(hf, dsa_long_context=True, max_seq_len=8192)
    assert long.max_seq_len == 8192 and long.dsa_long_context
    with pytest.raises(ValueError, match="index_topk"):
        KimiK3ModelConfig.k31(max_seq_len=4096)
    with pytest.raises(ValueError, match="DSA checkpoint"):
        KimiK3ModelConfig.k3(dsa_long_context=True)


def test_index_skip_topk_offset_skips_no_k31_layer():
    """K3.1's ``index_skip_topk_offset`` (1 in the release) under DeepSeek-V3.2's IndexCache rule, as
    vLLM reads it: a layer reuses an earlier top-k when ``max(layer - offset + 1, 0) % freq != 0``. At the
    release's ``index_topk_freq`` 1 every DSA layer runs its own indexer whatever the offset, so each MLA
    layer keeps an index plane; another frequency is refused (vLLM's K3.1 refuses it too)."""
    from mstar.model.kimi_k3.config import index_skips_topk

    hf = _hf("k31_config.json")
    dsa = hf["text_config"]["dsa_config"]
    assert (dsa["index_topk_freq"], dsa["index_skip_topk_offset"]) == (1, 1)
    full = KimiK3ModelConfig.from_hf_config(hf).full_attn_layers
    for offset in (0, 1, 2, 7):
        assert not any(index_skips_topk(layer, 1, offset) for layer in full)
        edited = copy.deepcopy(hf)
        edited["text_config"]["dsa_config"]["index_skip_topk_offset"] = offset
        assert KimiK3ModelConfig.from_hf_config(edited) == KimiK3ModelConfig.k31()
    # the rule at a frequency K3.1 doesn't use: offset 1 keeps every 4th layer from 0, offset 2 from 1
    assert [layer for layer in range(9) if not index_skips_topk(layer, 4, 1)] == [0, 4, 8]
    assert [layer for layer in range(9) if not index_skips_topk(layer, 4, 2)] == [0, 1, 5]


@pytest.mark.parametrize("edit, match", [
    (lambda t: t["dsa_config"].update(index_topk_freq=4), "index_topk_freq"),
    (lambda t: t["dsa_config"].update(index_rope_dim=64), "rotary DSA"),
    (lambda t: t["dsa_config"].update(index_k_norm_type="rmsnorm"), "index_k_norm_type"),
    (lambda t: t.update(hidden_act="silu"), "SiTU"),
    (lambda t: t.update(num_expert_group=8), "expert-group"),
    (lambda t: t.update(mla_use_nope=False), "NoPE"),
    (lambda t: t["quantization_config"].update(format="nvfp4-pack-quantized"), "mxfp4-pack-quantized"),
    (lambda t: t.update(num_hidden_layers=94), "partition"),
])
def test_unported_layouts_are_refused(edit, match):
    hf = copy.deepcopy(_hf("k31_config.json"))
    edit(hf["text_config"])
    with pytest.raises(ValueError, match=match):
        KimiK3ModelConfig.from_hf_config(hf, strict=False)


def test_strict_parse_pins_the_release_geometry():
    hf = copy.deepcopy(_hf("k31_config.json"))
    hf["text_config"]["moe_intermediate_size"] = 2048
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        KimiK3ModelConfig.from_hf_config(hf)
    assert KimiK3ModelConfig.from_hf_config(hf, strict=False).moe_intermediate_size == 2048
