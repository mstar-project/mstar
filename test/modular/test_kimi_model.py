import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
)
from mstar.engine.resources import (
    AttentionSpec, KVSpec, PositionSpec, SamplerSpec, SamplingReqConfig,
)
from mstar.graph.base import Loop
from mstar.model.kimi_k2_7.config import SAMPLER, KimiK2Config
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model
from mstar.model.kimi_k2_7.submodules import KimiLLMSubmodule


def _make_model() -> KimiK2Model:
    model = object.__new__(KimiK2Model)
    model.config = KimiK2Config.reduced()
    model._submodule_cache = {}
    return model


def test_kimi_graph_walks():
    model = _make_model()

    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"}
    assert isinstance(walks["decode"], Loop)
    assert walks["decode"].name == "decode_loop"


def test_kimi_declares_one_resource_of_each_kind_on_the_llm_node():
    specs = _make_model().get_node_resources()
    assert [type(spec) for spec in specs] == [
        KVSpec, AttentionSpec, SamplerSpec, PositionSpec
    ]
    assert all(spec.nodes == {"LLM"} for spec in specs)


def test_kimi_kv_cache_config_matches_reduced_mla_dims():
    model = _make_model()
    cfg = model.config

    kv, _attn = model._kv_and_attn_specs()

    assert kv.num_layers == cfg.num_hidden_layers == 2
    assert kv.num_kv_heads == cfg.num_attention_heads == 4
    assert kv.num_qo_heads == cfg.num_attention_heads == 4
    # FlashInfer-SM90 requires padded_head_dim, not raw qk_head_dim.
    assert cfg.qk_head_dim == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim == 24
    assert kv.head_dim == cfg.padded_head_dim == 64
    assert kv.max_seq_len == cfg.max_position_embeddings


def test_kimi_prefill_transitions_to_decode():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["text"],
        graph_walk="prefill",
        is_prefill=True,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={"new_token": []},
    )

    assert result.full_metadata.graph_walk == "decode"
    assert result.full_metadata.is_prefill is False
    assert result.step_metadata["is_prefill"] is False
    assert result.request_done is False


def test_kimi_decode_completion_marks_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["text"],
        graph_walk="decode",
        is_prefill=False,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={},
    )

    assert result.request_done is True
    assert result.full_metadata.kwargs["decode_finished"] is True


def test_kimi_get_submodule_is_dummy_mode():
    model = _make_model()
    assert getattr(model, "model_path_hf", None) is None
    assert model.get_submodule("LLM") is None


def test_check_stop_matches_any_token_in_eos_token_ids():
    config = KimiK2Config.reduced()
    config.eos_token_ids = [5, 9]
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    submodule = KimiLLMSubmodule(language_model=language_model, config=config)

    info = CurrentForwardPassInfo(
        request_id="r0", graph_walk="decode", fwd_index=0, random_seed=0,
        max_tokens=100,
        resource_configs={SAMPLER: SamplingReqConfig(ignore_eos=False)},
    )

    assert submodule.check_stop(
        "r0", info, {"new_token": [torch.tensor([5])]}
    ) == {"decode_loop"}
    assert submodule.check_stop(
        "r0", info, {"new_token": [torch.tensor([9])]}
    ) == {"decode_loop"}
    assert submodule.check_stop(
        "r0", info, {"new_token": [torch.tensor([7])]}
    ) == set()


# --- process_prompt chat-template path (Phase B) ---------------------------


class _StubTokenizer:
    """Records apply_chat_template / plain-call args; no HF IO."""

    def __init__(self, encoded):
        self._encoded = encoded
        self.chat_calls = []
        self.plain_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return self._encoded

    def __call__(self, prompt, return_tensors=None):
        self.plain_calls.append((prompt, return_tensors))

        class _Enc:
            input_ids = torch.tensor([[1, 2, 3]])

        return _Enc()


def _make_chat_model(tokenizer, tokenizer_mode="hf") -> KimiK2Model:
    model = _make_model()
    model._tokenizer_mode = tokenizer_mode
    model._tokenizer = tokenizer
    return model


def test_process_prompt_messages_tensor_return():
    tok = _StubTokenizer(torch.tensor([[1, 2, 3]]))
    model = _make_chat_model(tok)
    messages = [{"role": "user", "content": "hi"}]

    result = model.process_prompt(
        None, ["text"], ["text"],
        messages=messages, tools=[{"type": "function"}],
        chat_template_kwargs={"k": "v"},
    )

    assert len(tok.chat_calls) == 1
    called_messages, kwargs = tok.chat_calls[0]
    assert called_messages == messages
    assert kwargs["tools"] == [{"type": "function"}]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is True
    assert kwargs["return_tensors"] == "pt"
    assert kwargs["k"] == "v"

    ids = result["text_inputs"][0]
    assert torch.equal(ids, torch.tensor([1, 2, 3]))
    assert ids.dtype == torch.long
    assert ids.ndim == 1


def test_process_prompt_messages_dict_return_shape():
    tok = _StubTokenizer({"input_ids": torch.tensor([[4, 5]])})
    model = _make_chat_model(tok)

    result = model.process_prompt(
        None, ["text"], ["text"], messages=[{"role": "user", "content": "hi"}],
    )

    ids = result["text_inputs"][0]
    assert torch.equal(ids, torch.tensor([4, 5]))
    assert ids.dtype == torch.long
    assert ids.ndim == 1


def test_process_prompt_tool_choice_none_drops_tools():
    tok = _StubTokenizer(torch.tensor([[1]]))
    model = _make_chat_model(tok)

    model.process_prompt(
        None, ["text"], ["text"],
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}], tool_choice="none",
    )

    _, kwargs = tok.chat_calls[0]
    assert kwargs["tools"] is None


def test_process_prompt_plain_string_path_unchanged():
    tok = _StubTokenizer(torch.tensor([[9]]))
    model = _make_chat_model(tok)

    result = model.process_prompt("hello", ["text"], ["text"])

    assert len(tok.chat_calls) == 0
    assert len(tok.plain_calls) == 1
    called_prompt, return_tensors = tok.plain_calls[0]
    assert called_prompt == "hello"
    assert return_tensors == "pt"
    assert torch.equal(result["text_inputs"][0], torch.tensor([1, 2, 3]))


def test_process_prompt_no_prompt_no_messages_returns_empty():
    model = _make_chat_model(_StubTokenizer(torch.tensor([[1]])))
    assert model.process_prompt(None, ["text"], ["text"]) == {}


def test_process_prompt_byte_mode_with_messages_raises():
    model = _make_chat_model(_StubTokenizer(torch.tensor([[1]])), tokenizer_mode="byte")
    with pytest.raises(ValueError, match="byte"):
        model.process_prompt(
            None, ["text"], ["text"], messages=[{"role": "user", "content": "hi"}],
        )


# --- postprocess: special tokens kept, eos ids dropped ----------------------


class _DecodeStubTokenizer:
    """decode() echoes the ids as a marker string so the test can see exactly
    what postprocess passed through (special tokens included)."""

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(f"<{i}>" for i in ids)


def _make_decode_model() -> KimiK2Model:
    model = _make_model()
    model._tokenizer_mode = "hf"
    model._tokenizer = _DecodeStubTokenizer()
    return model


def test_postprocess_keeps_special_tokens_and_drops_eos_ids():
    model = _make_decode_model()
    model.config.eos_token_ids = [9]

    out = model.postprocess(torch.tensor([9, 1, 2]), "text")

    assert out == "<1><2>".encode("utf-8")


def test_postprocess_byte_mode_unchanged():
    model = _make_model()
    model._tokenizer_mode = "byte"
    out = model.postprocess(torch.tensor([65, 66]), "text")
    assert out == b"AB"
