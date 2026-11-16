"""Text-node contracts without weights, resource allocation or a GPU runtime."""

from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import FINAL_PREFILL
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    apply_yaml_overrides,
    resolve_spec_dependencies,
)
from mstar.model.keys import KV_CACHE, MIXED, PREFILL_TEXT, ROPE, SAMPLER
from mstar.model.ministral3.config import ATTN, LLM_NODE, Ministral3TextConfig
from mstar.model.ministral3.resources import text_node_resources
from mstar.model.ministral3.submodules import Ministral3Decoder
from mstar.model.multimodal import media_walks, text_combined_walks
from mstar.model.submodule_base import END_PARTITION, ARNodeInputs, HostRows


@pytest.fixture
def node():
    # Real 3B shapes, no 3B allocation. Stop ID is synthetic, not a tokenizer claim.
    with torch.device("meta"):
        return Ministral3Decoder(Ministral3TextConfig(stop_token_ids=(2,)))


def test_resources_match_projection_geometry_and_support_deployment_overrides(node):
    specs = text_node_resources(node.config)
    resources = resolve_spec_dependencies(specs)
    assert set(resources) == {KV_CACHE, ATTN, ROPE, SAMPLER}
    assert all(spec.nodes == {LLM_NODE} for spec in specs)
    kv = resources[KV_CACHE].config
    assert (kv.num_layers, kv.num_qo_heads, kv.num_kv_heads, kv.head_dim) == (26, 32, 8, 128)
    attn = node.model.layers[0].self_attn
    assert attn.qkv_proj.weight.shape == ((kv.num_qo_heads + 2 * kv.num_kv_heads) * kv.head_dim, 3072)
    assert attn.o_proj.weight.shape == (3072, kv.num_qo_heads * kv.head_dim)
    pos = resources[ROPE].config
    assert pos.rotary_dim == kv.head_dim
    assert pos.yarn == node.config.yarn
    assert pos.rope_theta == node.config.rope_theta
    assert resources[ATTN].depends_on() == resources[ROPE].depends_on() == {KV_CACHE}
    assert resources[ATTN].config.sliding_window is None
    assert resources[SAMPLER].vocab_size == node.model.lm_head.weight.shape[0]
    apply_yaml_overrides(specs, {"resources": {KV_CACHE: {"max_num_pages": 32, "page_size": 16}}})
    assert (kv.max_num_pages, kv.page_size) == (32, 16)
    assert kv.max_seq_len == node.config.max_position_embeddings
    assert pos.yarn == node.config.yarn  # capacity tuning must not change the rotation


@pytest.mark.parametrize("walk,spans", [(PREFILL_TEXT, [3, 5]), ("decode", [1, 1]), (MIXED, [1, 5])])
def test_steps_keep_request_spans_and_resource_keys_aligned(node, walk, spans):
    inputs = [ARNodeInputs(input_seq_len=n, input_ids=torch.arange(n),
                          graph_walk=("decode" if i == 0 else PREFILL_TEXT) if walk == MIXED else walk)
              for i, n in enumerate(spans)]
    step = node.declare_step(walk, [17, 4], inputs)
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [(17, "main", spans[0]), (4, "main", spans[1])]
    specs = resolve_spec_dependencies(text_node_resources(node.config))
    assert step.steps.keys() == specs.keys()
    assert isinstance(step.steps[KV_CACHE], KVStep)
    assert isinstance(step.steps[ATTN], AttentionStep) and step.steps[ATTN].causal
    assert step.steps[ROPE] == PositionStep(segments=step.segments)  # engine derives positions and advance
    sampler = step.steps[SAMPLER]
    assert isinstance(sampler, SamplerStep) and sampler.apply_penalty
    expected = {17, 4} if walk == PREFILL_TEXT else {4} if walk == MIXED else set()
    assert sampler.prefill_tracked_tokens.keys() == expected


def test_walk_capture_and_token_feedback_contracts_agree(node, monkeypatch):
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    node.node_resources[SAMPLER] = SimpleNamespace(has_slot_masters=True)
    walks = media_walks(node.config.max_position_embeddings)
    assert set(walks) == {node.PREFILL_WALK, node.DECODE_WALK}
    assert walks[node.DECODE_WALK].name == node.DECODE_LOOP
    assert text_combined_walks() == {LLM_NODE: {MIXED: set(walks)}}
    configs = node.get_accelerator_graph_configs(torch.device("cpu"))
    assert {config.capture_graph_walk for config in configs} == {*walks, MIXED}
    # This checks declarations, not that a GPU capture can run or replay.
    prefill = node.prepare_inputs(PREFILL_TEXT, None, {"text_inputs": [torch.arange(5)]})
    chunk = node.split_inputs(PREFILL_TEXT, None, prefill, 2, 5)
    assert chunk.input_seq_len == 3
    assert chunk.input_ids.tolist() == [2, 3, 4]
    decode = node.prepare_inputs("decode", None, {}, has_loopback=True)
    assert decode.input_seq_len == 1 and decode.input_ids is None
    assert node.device_loopback.sampler == SAMPLER
    assert node.device_loopback.buffer == node.token_stop.buffer == "new_token"


@pytest.mark.parametrize("final,token,budget,expected", [
    (False, 2, 1, {}),  # a non-final prompt chunk does not finish the reply
    (True, 2, 2, {17: {END_PARTITION}}),
    (True, 9, 1, {17: {END_PARTITION}}),
    (True, 9, 2, {}),
])
def test_prefill_stops_only_on_a_reply_token(node, final, token, budget, expected):
    info = SimpleNamespace(graph_walk=PREFILL_TEXT, max_tokens=budget,
                           resource_configs={SAMPLER: SimpleNamespace(ignore_eos=False)},
                           step_metadata={FINAL_PREFILL: final})
    rows = HostRows(request_ids=(17,), buffers={"new_token": torch.tensor([[token]])})
    assert node.check_stop_batched([17], {17: info}, rows) == expected


def test_text_node_refuses_media(node):
    with pytest.raises(ValueError, match="unexpected inputs"):
        node.prepare_inputs(PREFILL_TEXT, None, {"text_inputs": [torch.arange(3)], "image": [torch.zeros(1)]})
    with pytest.raises(ValueError, match="does not support walk"):
        node.prepare_inputs("prefill_vision", None, {"text_inputs": [torch.arange(3)]})


@pytest.mark.parametrize("stops", [(), (-1,), (131072,), (True,)])
def test_stop_ids_must_be_explicit_vocabulary_ids(stops):
    with pytest.raises(ValueError, match="stop_token_ids"):
        Ministral3TextConfig(stop_token_ids=stops)
