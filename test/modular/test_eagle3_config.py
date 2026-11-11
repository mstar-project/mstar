"""EAGLE-3's checkpoint configs (both formats), its refusals, its ``k`` and its declaration. The drafter's forward
is checked end to end on the GPU, against RedHatAI/Qwen3-8B-speculator.eagle3."""
from __future__ import annotations

import json
import logging

import pytest

from mstar.engine.resources.draft.config import DraftConfig
from mstar.engine.resources.draft.drafter import ChunkBoundary
from mstar.model.components.eagle3 import Eagle3, Eagle3Config, default_taps

# RedHatAI/Qwen3-8B-speculator.eagle3's config.json (the speculators format), its keys that matter
SPECULATORS = {
    "architectures": ["Eagle3Speculator"], "draft_vocab_size": 32000, "norm_before_residual": True,
    "speculators_config": {"algorithm": "eagle3", "proposal_methods": [
        {"proposal_type": "greedy", "speculative_tokens": 3}], "verifier": {"name_or_path": "Qwen/Qwen3-8B"}},
    "speculators_model_type": "eagle3", "target_hidden_size": None, "torch_dtype": "bfloat16",
    "transformer_layer_config": {
        "attention_bias": False, "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 12288,
        "mlp_bias": False, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 1,
        "num_key_value_heads": 8, "rms_norm_eps": 1e-6, "rope_theta": 1000000, "vocab_size": 151936,
    },
}
# the EAGLE authors' Llama-style format
LLAMA = {
    "architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 32000, "hidden_size": 4096,
    "intermediate_size": 12288, "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 1,
    "rms_norm_eps": 1e-6, "rope_theta": 1000000, "vocab_size": 151936,
    "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 17, 32]},
}


def test_the_speculators_and_llama_formats_parse_to_one_config():
    spec = Eagle3Config.from_hf(SPECULATORS)
    assert (spec.hidden_size, spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim) == (4096, 32, 8, 128)
    assert (spec.vocab_size, spec.draft_vocab_size, spec.target_hidden_size) == (151936, 32000, 4096)
    assert spec.norm_before_residual and spec.trained_depth == 3 and spec.taps is None
    assert spec.layer_prefix == "layers.0." and spec.rope_theta == 1e6
    llama = Eagle3Config.from_hf(LLAMA)
    assert llama.layer_prefix == "midlayer." and llama.taps == (1, 17, 32) and llama.trained_depth is None
    assert not llama.norm_before_residual and llama.head_dim == 128
    # Qwen3-8B's 36 layers: the inputs of 2, 18 and 33
    assert default_taps(36) == (2, 18, 33)


@pytest.mark.parametrize("change, why", [
    ({"num_hidden_layers": 2}, "2 decoder layers"),
    ({"norm_before_fc": True}, "norm_before_fc"),
    ({"fc_norm": True}, "fc_norm"),
    ({"norm_output": True}, "norm_output"),
    ({"parallel_drafting": True}, "parallel_drafting"),
    ({"attention_bias": True}, "biases"),
])
def test_the_variants_no_reference_pair_uses_are_refused(change, why):
    hf = {**SPECULATORS, "transformer_layer_config": {**SPECULATORS["transformer_layer_config"], **change}}
    with pytest.raises(ValueError, match=why):
        Eagle3Config.from_hf(hf)
    with pytest.raises(ValueError, match=why):
        Eagle3Config.from_hf({**LLAMA, **change})
    with pytest.raises(ValueError, match="not an EAGLE-3 checkpoint"):
        Eagle3Config.from_hf({"model_type": "llama"})


def test_k_defaults_to_three_runs_to_seven_and_warns_past_the_trained_depth(caplog):
    assert DraftConfig.parse({"method": "eagle3", "model": "m"}).num_draft_tokens == 3
    with pytest.raises(ValueError, match="at most 7"):
        DraftConfig.parse({"method": "eagle3", "model": "m", "num_draft_tokens": 8})
    config = Eagle3Config.from_hf(SPECULATORS)
    with caplog.at_level(logging.WARNING):
        Eagle3(config, 3, "bfloat16", taps=(2, 18, 33), kv_key="kv_cache", pos_key="rope", attn_key="attn")
    assert not caplog.records
    with caplog.at_level(logging.WARNING):
        Eagle3(config, 7, "bfloat16", taps=(2, 18, 33), kv_key="kv_cache", pos_key="rope", attn_key="attn")
    assert "past the 3 its checkpoint was trained" in caplog.text


def test_its_declaration_reads_three_taps_carries_its_prenorm_and_joins_one_layer():
    decl = Eagle3(Eagle3Config.from_hf(SPECULATORS), 3, "bfloat16", taps=(2, 18, 33), kv_key="kv_cache",
                  pos_key="rope", attn_key="attn")
    needs = decl.needs
    assert needs.taps == (2, 18, 33) and needs.lag is None and not needs.target_hidden
    assert (needs.kv.num_layers, needs.kv.num_kv_heads, needs.kv.head_dim) == (1, 8, 128)
    assert needs.carry.width == 4096 and needs.chunk_boundary == ChunkBoundary.UNDECLARED
    # so a target that prefills prompts in chunks refuses it at startup, until one has a next-ids input for it
    from types import SimpleNamespace

    from mstar.engine.resources.draft.startup import chunking_problem

    node = SimpleNamespace(node="LLM", needs=needs)
    assert "no chunk-boundary rule" in chunking_problem(node, chunks_prompts=True)
    assert chunking_problem(node, chunks_prompts=False) is None
    assert needs.geometry.hidden_size == 4096 and needs.geometry.vocab_size == 151936
    (taps,) = decl.node_resources({"LLM"})
    assert taps.layers == (2, 18, 33) and taps.hidden_size == 4096


# RedHatAI/Qwen3-8B-speculator.eagle3's model.safetensors header (name -> shape), as read on the GPU node
SPECULATOR_SHAPES = {
    "d2t": [32000], "embed_tokens.weight": [151936, 4096], "fc.weight": [4096, 12288],
    "layers.0.hidden_norm.weight": [4096], "layers.0.input_layernorm.weight": [4096],
    "layers.0.mlp.down_proj.weight": [4096, 12288], "layers.0.mlp.gate_proj.weight": [12288, 4096],
    "layers.0.mlp.up_proj.weight": [12288, 4096], "layers.0.post_attention_layernorm.weight": [4096],
    "layers.0.self_attn.k_proj.weight": [1024, 8192], "layers.0.self_attn.o_proj.weight": [4096, 4096],
    "layers.0.self_attn.q_proj.weight": [4096, 8192], "layers.0.self_attn.v_proj.weight": [1024, 8192],
    "lm_head.weight": [32000, 4096], "norm.weight": [4096], "t2d": [151936],
}


def _cached_header() -> dict | None:
    """The cached checkpoint's own header, when this machine has the snapshot (no download)."""
    import json
    import struct
    from pathlib import Path

    for root in (Path.home() / ".cache/huggingface/hub",):
        for path in root.glob("models--RedHatAI--Qwen3-8B-speculator.eagle3/snapshots/*/model.safetensors"):
            with open(path, "rb") as f:
                header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    return None


def test_every_drafter_parameter_has_the_checkpoints_shape():
    """Built on the meta device from the speculators config, each parameter is the shape the checkpoint ships
    (q/k/v and gate/up stacked as the loader stacks them): the attention reads 2H and writes H back."""
    import torch

    from mstar.model.components.eagle3 import Eagle3Drafter

    shapes = _cached_header() or SPECULATOR_SHAPES
    decl = Eagle3(Eagle3Config.from_hf(SPECULATORS), 3, "bfloat16", taps=(2, 18, 33), kv_key="kv_cache",
                  pos_key="rope", attn_key="attn")
    with torch.device("meta"):
        drafter = Eagle3Drafter(decl)
    layer = {k.removeprefix("layers.0."): v for k, v in shapes.items()}
    stacked = {
        "self_attn.qkv_proj.weight": [sum(layer[f"self_attn.{p}_proj.weight"][0] for p in "qkv"), 8192],
        "mlp.gate_up_proj.weight": [2 * layer["mlp.gate_proj.weight"][0], 4096],
    }
    for name, param in drafter.named_parameters():
        want = stacked.get(name) or layer[name]
        assert list(param.shape) == want, (name, list(param.shape), want)
    assert sorted(name for name, _ in drafter.named_parameters()) == sorted(
        [k for k in layer if k not in ("d2t", "t2d") and not k.startswith(("self_attn.q_", "self_attn.k_",
                                                                          "self_attn.v_", "mlp.gate_", "mlp.up_"))]
        + list(stacked))


def test_a_speculators_checkpoints_top_level_dtype_is_the_drafters(tmp_path):
    """The format names it at the top, not in its layer config; the startup check compares this to the target's."""
    from qwen3_cpu import make_model, write_checkpoints

    target, _ = write_checkpoints(tmp_path)
    draft = tmp_path / "eagle3"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({**SPECULATORS, "torch_dtype": "float32"}))
    model = make_model(target, {"method": "eagle3", "model": str(draft), "num_draft_tokens": 3})
    assert model.drafter_decl.dtype == "float32"


def test_norm_before_residual_defaults_to_each_formats_reference():
    """Absent from the config: the speculators library's default (True), the EAGLE authors' (False)."""
    spec = {k: v for k, v in SPECULATORS.items() if k != "norm_before_residual"}
    llama = {k: v for k, v in LLAMA.items() if k != "norm_before_residual"}
    assert Eagle3Config.from_hf(spec).norm_before_residual is True
    assert Eagle3Config.from_hf(llama).norm_before_residual is False
    assert Eagle3Config.from_hf({**spec, "norm_before_residual": False}).norm_before_residual is False


def _tiny_drafter():
    from mstar.model.components.eagle3 import Eagle3Drafter

    layer = {**SPECULATORS["transformer_layer_config"], "hidden_size": 64, "intermediate_size": 128,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 96}
    config = Eagle3Config.from_hf({**SPECULATORS, "draft_vocab_size": 32, "transformer_layer_config": layer})
    decl = Eagle3(config, 3, "float32", taps=(0, 1, 2), kv_key="kv_cache", pos_key="rope", attn_key="attn")
    return Eagle3Drafter(decl)


def test_a_checkpoint_without_an_embedding_shares_the_targets_by_reference(monkeypatch):
    import torch

    from mstar.model import loader

    drafter = _tiny_drafter()
    # every parameter but the embedding came from the checkpoint
    loaded = {name for name, _ in drafter.named_parameters() if not name.startswith("embed_tokens.")}
    monkeypatch.setattr(loader, "load_hf_weights", lambda module, weights, **kw: set(loaded))
    target = _tiny_drafter()
    drafter.load_weights([("d2t", torch.zeros(32))], target=target)
    assert drafter.embed_tokens is target.embed_tokens
    other = _tiny_drafter()
    other.embed_tokens.weight = torch.nn.Parameter(torch.zeros(48, 64))
    with pytest.raises(ValueError, match="embedding"):
        _tiny_drafter().load_weights([("d2t", torch.zeros(32))], target=other)


def test_the_built_node_drafts_with_its_eagle3_module(tmp_path):
    """The EAGLE-3 drafter is an nn.Module: it lives in the node's ``_modules``, and nothing at
    class level may shadow it (the engine refuses a verifying node whose ``drafter`` reads None)."""
    from qwen3_cpu import bind_share, build_submodule, make_model, write_checkpoints

    from mstar.model.components.eagle3 import Eagle3Drafter

    target, _ = write_checkpoints(tmp_path)
    draft = tmp_path / "eagle3"
    draft.mkdir()
    # the CPU target's shapes (qwen3_cpu.TARGET), over a 32-token draft vocabulary
    layer = {**SPECULATORS["transformer_layer_config"], "hidden_size": 32, "intermediate_size": 48,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 97,
             "rope_theta": 10_000.0}
    (draft / "config.json").write_text(json.dumps(
        {**SPECULATORS, "draft_vocab_size": 32, "torch_dtype": "float32", "transformer_layer_config": layer}))
    model = make_model(target, {"method": "eagle3", "model": str(draft), "num_draft_tokens": 3}, max_seq_len=512)
    sub = build_submodule(model)
    assert isinstance(sub.drafter, Eagle3Drafter) and sub.drafter is sub._modules["drafter"]
    # k is the engine's to bind at load
    assert bind_share(sub, 3)._block == 4
