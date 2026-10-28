"""The generic text LM (``mstar/model/text_lm``) on CPU: checkpoint config to resources, the
prefill -> decode state machine, prompt checks and stop handling, without weights."""

import json
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import AttentionSpec, KVSpec, PositionSpec, SamplerSpec, SamplingReqConfig
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.model.submodule_base import HostRows
from mstar.model.text_lm.config import DECODE_LOOP, DECODE_WALK, LLM_NODE, PREFILL_WALK, SAMPLER
from mstar.model.text_lm.submodules import TextLMSubmodule
from mstar.model.text_lm.text_lm_model import TextLMModel

# Qwen3-Coder-30B-A3B-Instruct's config.json, layers cut to 2
CODER = {
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "attention_bias": False,
    "bos_token_id": 151643, "decoder_sparse_step": 1, "eos_token_id": 151645, "head_dim": 128,
    "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 6144, "max_position_embeddings": 262144,
    "mlp_only_layers": [], "moe_intermediate_size": 768, "norm_topk_prob": True, "num_attention_heads": 32,
    "num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 2, "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 10000000, "sliding_window": None,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16", "use_sliding_window": False, "vocab_size": 151936,
}
GENERATION = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05,
              "eos_token_id": [151645, 151643]}
IM_END, ENDOFTEXT, WORD = 151645, 151643, 1234


def _checkpoint(tmp_path, config=None, generation=GENERATION):
    (tmp_path / "config.json").write_text(json.dumps(config or CODER))
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    return str(tmp_path)


def _model(tmp_path, **kwargs) -> TextLMModel:
    return TextLMModel(model_path_hf=_checkpoint(tmp_path), **kwargs)


def test_registry_cli_and_adapter_agree():
    from mstar.api_server.openai.adapters import TextChatAdapter, get_adapter
    from mstar.cli.main import DEFAULT_CONFIGS

    assert get_model_class("qwen3_coder_30b_a3b") is TextLMModel
    assert HF_MODELS["qwen3_coder_30b_a3b"]["model_path_hf"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert DEFAULT_CONFIGS["qwen3_coder_30b_a3b"] == "qwen3_coder_30b_a3b.yaml"
    assert isinstance(get_adapter("qwen3_coder_30b_a3b"), TextChatAdapter)
    # Qwen3 dense, on the same model
    for size in ("0.6", "1.7", "4", "8"):
        name = f"qwen3_{size}b"
        assert get_model_class(name) is TextLMModel
        assert HF_MODELS[name]["model_path_hf"] == f"Qwen/Qwen3-{size}B"
        assert DEFAULT_CONFIGS[name] == f"{name}.yaml"
        assert isinstance(get_adapter(name), TextChatAdapter)


def test_resources_follow_the_checkpoint(tmp_path):
    model = _model(tmp_path, max_seq_len=4096)
    specs = {type(s): s for s in model.get_node_resources()}
    kv = specs[KVSpec].config
    assert (kv.num_layers, kv.num_kv_heads, kv.num_qo_heads, kv.head_dim, kv.max_seq_len) == (2, 4, 32, 128, 4096)
    assert specs[PositionSpec].config.rope_theta == 1e7
    assert specs[SamplerSpec].vocab_size == 151936
    assert all(s.nodes == {LLM_NODE} for s in specs.values()) and AttentionSpec in specs
    (label,) = model.kv_labels()["kv_cache"].values()
    assert label.keys.walks == (PREFILL_WALK,) and label.keys.growth
    # no weights: dummy mode
    assert model.get_submodule(LLM_NODE) is None


def test_a_dense_qwen3_checkpoint_is_served_too(tmp_path):
    dense = {k: v for k, v in CODER.items() if k not in ("num_experts", "num_experts_per_tok",
                                                         "moe_intermediate_size", "decoder_sparse_step")}
    model = TextLMModel(model_path_hf=_checkpoint(tmp_path, {**dense, "model_type": "qwen3"}))
    assert model.config.num_experts == 0 and not model.prebuild_fused_moe
    assert _model(tmp_path).prebuild_fused_moe


@pytest.mark.parametrize("change,match", [
    ({"model_type": "llama"}, "model_type"),
    ({"rope_scaling": {"type": "yarn", "factor": 4.0}}, "rope_scaling"),
    ({"use_sliding_window": True}, "sliding"),
    ({"attention_bias": True}, "attention_bias"),
])
def test_what_the_decoder_does_not_implement_is_refused(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        TextLMModel(model_path_hf=_checkpoint(tmp_path, {**CODER, **change}))


def test_stop_ids_are_every_eos_the_configs_name(tmp_path):
    # no tokenizer files here; with them, the tokenizer's EOS joins
    assert _model(tmp_path).stop_token_ids == frozenset({IM_END, ENDOFTEXT})
    bare = tmp_path / "bare"
    bare.mkdir()
    assert TextLMModel(model_path_hf=_checkpoint(bare, generation=None)).stop_token_ids == frozenset({IM_END})


def test_sampling_defaults_come_from_generation_config(tmp_path):
    model = _model(tmp_path)
    cfg = model.get_request_resource_configs({}, {})[SAMPLER]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.repetition_penalty) == (0.7, 0.8, 20, 1.05)
    cfg = model.get_request_resource_configs({}, {"temperature": 0.0, "top_k": 0})[SAMPLER]
    assert (cfg.temperature, cfg.top_k, cfg.top_p) == (0.0, 0, 0.8)
    greedy = TextLMModel(model_path_hf=_checkpoint(tmp_path, generation={"do_sample": False, "top_k": 20}))
    cfg = greedy.get_request_resource_configs({}, {})[SAMPLER]
    assert (cfg.temperature, cfg.top_k) == (0.0, 0)


def test_state_machine_prefill_then_decode_then_done(tmp_path):
    model = _model(tmp_path)
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {PREFILL_WALK, DECODE_WALK}
    assert walks[DECODE_WALK].max_iters == model.max_seq_len - 1
    args = model.get_initial_forward_pass_args("default", ["text"], ["text"], {"text_inputs": [object()]})
    assert args.full_metadata.graph_walk == PREFILL_WALK and args.full_metadata.is_prefill
    assert [(e.next_node, e.name) for e in args.inputs] == [(LLM_NODE, "text_inputs")]
    nxt = model.get_partition_forward_pass_args("default", args.full_metadata, {"new_token": [object()]})
    assert nxt.full_metadata.graph_walk == DECODE_WALK and len(nxt.inputs[0].tensor_info) == 1
    assert model.get_partition_forward_pass_args("default", nxt.full_metadata, {}).request_done
    assert model.get_loop_caps(max_output_tokens=1) == {DECODE_LOOP: 1}
    assert model.get_loop_caps(max_output_tokens=10) == {DECODE_LOOP: 9}
    assert model.get_loop_caps() == {DECODE_LOOP: model.default_max_output_tokens - 1}


def test_prompt_ids_are_checked_against_the_context(tmp_path):
    model = _model(tmp_path, max_seq_len=64)
    ids = torch.arange(10)
    out = model.process_prompt(None, ["text"], ["text"], tensors={"text_inputs": [ids]}, max_output_tokens=54)
    assert torch.equal(out.new_input_tensors["text_inputs"][0], ids)
    (span,) = out.metadata["kv_layout"]["kv_cache"]["main"].spans
    assert (span.kind, span.walk, span.length) == ("ids", PREFILL_WALK, 10)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.process_prompt(None, ["text"], ["text"], tensors={"text_inputs": [ids]}, max_output_tokens=55)
    with pytest.raises(ValueError, match="vocabulary"):
        model.process_prompt(None, ["text"], ["text"], tensors={"text_inputs": [torch.tensor([151936])]},
                             max_output_tokens=1)
    with pytest.raises(ValueError, match="text"):
        model.process_prompt("hi", ["text", "image"], ["text"])


def _submodule():
    return TextLMSubmodule(torch.nn.Linear(1, 1), frozenset({IM_END, ENDOFTEXT}))


def _info(walk, decoded=0, max_tokens=4096, ignore_eos=False):
    return SimpleNamespace(
        graph_walk=walk, max_tokens=max_tokens, dynamic_loop_iter_counts={DECODE_LOOP: decoded},
        resource_configs={SAMPLER: SamplingReqConfig(ignore_eos=ignore_eos)},
    )


def _tok(t):
    return {"new_token": [torch.tensor([t])]}


def test_stop_tokens_and_budget():
    sub = _submodule()
    assert sub.check_stop("a", _info(DECODE_WALK), _tok(IM_END)) == {DECODE_LOOP}
    assert sub.check_stop("a", _info(DECODE_WALK), _tok(ENDOFTEXT)) == {DECODE_LOOP}
    assert sub.check_stop("a", _info(DECODE_WALK), _tok(WORD)) == set()
    assert sub.check_stop("a", _info(DECODE_WALK, ignore_eos=True), _tok(IM_END)) == set()
    # max_tokens counts the prefill's token: decode iteration k makes token k + 2
    assert sub.check_stop("a", _info(DECODE_WALK, decoded=7, max_tokens=10), _tok(WORD)) == set()
    assert sub.check_stop("a", _info(DECODE_WALK, decoded=8, max_tokens=10), _tok(WORD)) == {DECODE_LOOP}


def test_a_stop_at_prefill_ends_the_loop_at_its_first_step_without_a_token():
    sub = _submodule()
    assert sub.check_stop("a", _info(PREFILL_WALK), _tok(IM_END)) == set()
    outputs = _tok(WORD)
    sub.postprocess("a", _info(DECODE_WALK), outputs)
    assert "new_token" not in outputs and "text_inputs" in outputs
    assert sub.check_stop("a", _info(DECODE_WALK), outputs) == {DECODE_LOOP}
    # another request is untouched
    outputs = _tok(WORD)
    sub.postprocess("b", _info(DECODE_WALK), outputs)
    assert torch.equal(outputs["new_token"][0], outputs["text_inputs"][0])


def test_batched_stop_check_agrees_with_per_request():
    rows = ("im_end", "word", "ignored", "budget", "prefill")
    tokens = torch.tensor([IM_END, WORD, IM_END, WORD, WORD, 0])  # a padding row at the end
    infos = {
        "im_end": _info(DECODE_WALK), "word": _info(DECODE_WALK),
        "ignored": _info(DECODE_WALK, ignore_eos=True), "budget": _info(DECODE_WALK, 8, max_tokens=10),
        "prefill": _info(PREFILL_WALK),
    }
    batched = _submodule().check_stop_batched(list(reversed(rows)), infos, HostRows(rows, {"new_token": tokens}))
    single = _submodule()
    per_request = {
        rid: got for i, rid in enumerate(rows)
        if (got := single.check_stop(rid, infos[rid], {"new_token": [tokens[i:i + 1]]}))
    }
    assert batched == per_request == {"im_end": {DECODE_LOOP}, "budget": {DECODE_LOOP}}


def test_declared_step_tracks_the_prompt_for_the_penalty_only_at_prefill():
    from mstar.model.submodule_base import ARNodeInputs

    sub = _submodule()
    inputs = [ARNodeInputs(input_seq_len=3, input_ids=torch.arange(3)),
              ARNodeInputs(input_seq_len=2, input_ids=torch.arange(2))]
    step = sub.declare_step(PREFILL_WALK, ["a", "b"], inputs)
    assert [(s.request_id, s.span) for s in step.segments] == [("a", 3), ("b", 2)]
    assert set(step.steps[SAMPLER].prefill_tracked_tokens) == {"a", "b"}
    step = sub.declare_step(DECODE_WALK, ["a"], [ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([5]))])
    assert step.steps[SAMPLER].prefill_tracked_tokens == {}


# ── speculative decoding (model_kwargs.draft) ──

DENSE = {
    **{k: v for k, v in CODER.items() if k not in ("num_experts", "num_experts_per_tok", "moe_intermediate_size",
                                                   "decoder_sparse_step")},
    "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
}


def _drafter(tmp_path, **change):
    directory = tmp_path / "drafter"
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps({**DENSE, "num_hidden_layers": 1, "num_key_value_heads": 2,
                                                       **change}))
    return str(directory)


def _layouts(model, n=10, budget=8):
    out = model.process_prompt(None, ["text"], ["text"], tensors={"text_inputs": [torch.arange(n)]},
                               max_output_tokens=budget)
    return out.metadata["kv_layout"]


def test_prompt_lookup_drafting_declares_a_verifying_node(tmp_path):
    from mstar.engine.resources import NGRAM, DraftVerifySpec
    from mstar.engine.resources.draft_verify.ngram import NgramSpec
    from mstar.model.submodule_base import ARNodeInputs

    model = _model(tmp_path, max_seq_len=64, draft={"method": "ngram", "max_drafts": 3})
    specs = {type(s): s for s in model.get_node_resources()}
    assert DraftVerifySpec in specs and specs[NgramSpec].max_drafts == 3
    sampler = specs[SamplerSpec]
    assert sampler.draft_verify and sampler.enable_sampled_verify and not sampler.enable_repetion_penalty
    # a verifying walk keeps no prefix-cache keys
    (label,) = model.kv_labels()["kv_cache"].values()
    assert label.keys is None
    # the layout carries the verify rows' lookahead (2k + 1), and the context check counts it
    (layout,) = _layouts(model)["kv_cache"].values()
    assert layout.lookahead == 7
    with pytest.raises(ValueError, match="lookahead"):
        _layouts(model, n=10, budget=48)

    sub = TextLMSubmodule(torch.nn.Linear(1, 1), frozenset({IM_END}), draft=model.draft)
    assert sub.get_draft_policy(DECODE_WALK).proposer == NGRAM and sub.get_draft_policy(PREFILL_WALK) is None
    row = sub.prepare_inputs(DECODE_WALK, None, {"text_inputs": [torch.tensor([4, 5])]})
    assert (row.input_seq_len, row.input_ids.tolist()) == (4, [5])
    step = sub.declare_step(DECODE_WALK, ["a"], [row])
    assert step.steps["draft_verify"].drafts == 3 and step.steps[NGRAM].drafts == 3
    prefill = sub.declare_step(PREFILL_WALK, ["a"], [ARNodeInputs(input_seq_len=3, input_ids=torch.arange(3))])
    assert set(prefill.steps[NGRAM].history) == {"a"} and not prefill.steps[SAMPLER].apply_penalty


def test_a_draft_model_declares_its_own_resources_and_layout(tmp_path):
    from mstar.graph.special_destinations import EMPTY_DESTINATION
    from mstar.model.submodule_base import ARNodeInputs

    model = _model(tmp_path, max_seq_len=64, draft={"method": "draft_model", "model": _drafter(tmp_path),
                                                     "max_drafts": 4})
    keys = {s.resource_key: s for s in model.get_node_resources()}
    assert {"draft_verify", "draft_kv", "draft_attn", "draft_pos"} <= set(keys) and "ngram" not in keys
    # the drafter's geometry, not the target's
    assert (keys["draft_kv"].config.num_layers, keys["draft_kv"].config.num_kv_heads) == (1, 2)
    assert set(model.kv_labels()) == {"kv_cache", "draft_kv"}
    layouts = _layouts(model)
    (target,), (drafter,) = layouts["kv_cache"].values(), layouts["draft_kv"].values()
    assert target.lookahead == drafter.lookahead == 9
    # the drafter holds the prompt but its last token
    assert target.spans[0].length == 10 and drafter.spans[0].length == 9
    with pytest.raises(ValueError, match="at least 2"):
        _layouts(model, n=1)
    # the prefill hands the decode loop the prompt's last token and the sampled one
    prefill = model.get_graph_walk_graphs()[PREFILL_WALK]
    assert any(e.name == "decode_input" and e.next_node == EMPTY_DESTINATION and e.persist for e in prefill.outputs)
    args = model.get_initial_forward_pass_args("default", ["text"], ["text"], {"text_inputs": [object()]})
    nxt = model.get_partition_forward_pass_args("default", args.full_metadata, {"decode_input": [object()]})
    assert len(nxt.inputs[0].tensor_info) == 1
    # prefill runs eager: the drafter prefills beside the target
    assert model.prefill_token_buckets == []

    class Drafter:
        decl = model.draft_model

    sub = TextLMSubmodule(torch.nn.Linear(1, 1), frozenset({IM_END}), draft=model.draft, drafter=Drafter())
    row = sub.prepare_inputs(DECODE_WALK, None, {"text_inputs": [torch.tensor([4, 5])]})
    assert (row.input_seq_len, row.input_ids.tolist()) == (5, [4, 5])
    step = sub.declare_step(DECODE_WALK, ["a"], [row])
    assert {"draft_kv", "draft_attn", "draft_pos", "draft_verify"} <= set(step.steps)
    prefill = sub.declare_step(PREFILL_WALK, ["a"], [ARNodeInputs(input_seq_len=3, input_ids=torch.arange(3))])
    assert [s.span for s in prefill.steps["draft_kv"].segments] == [2]


def test_a_drafting_node_drops_the_checkpoints_default_penalty(tmp_path):
    """Qwen3-Coder's generation_config.json defaults repetition_penalty to 1.05, which a verify
    step can't apply: the default goes, so requests that name none are served, and a request
    that names one is refused (400) by the sampler's draft-verify check."""
    model = _model(tmp_path, draft={"method": "ngram"})
    config = model.get_request_resource_configs({}, {})[SAMPLER]
    assert config.repetition_penalty in (None, 1.0) and config.temperature == 0.7
    spec = next(s for s in model.get_node_resources() if isinstance(s, SamplerSpec))
    config.validate(spec)
    named = model.get_request_resource_configs({}, {"repetition_penalty": 1.2})[SAMPLER]
    with pytest.raises(ValueError, match="repetition_penalty"):
        named.validate(spec)
    # without drafting, the default still applies
    assert _model(tmp_path).get_request_resource_configs({}, {})[SAMPLER].repetition_penalty == 1.05


def test_a_draft_model_of_another_tokenizer_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tokenizes differently"):
        _model(tmp_path, draft={"method": "draft_model", "model": _drafter(tmp_path, vocab_size=1000)})
    with pytest.raises(ValueError, match="'llama'"):
        _model(tmp_path, draft={"method": "draft_model", "model": _drafter(tmp_path, model_type="llama")})
