"""The serving config's ``quantization:`` block: what it parses to, and what it refuses before anything loads."""

import pytest

from mstar.quantization.formats import AUTO, BF16, FORMATS
from mstar.quantization.spec import QuantizationSpecError, Rule, parse_quantization, pattern_matches, spec_problems


def test_absent_block_serves_the_checkpoint_as_stored():
    spec = parse_quantization(None)
    assert not spec.given
    assert spec.for_node("LLM").format == AUTO and spec.for_node("LLM").is_default


def test_short_form_is_one_format_for_every_node():
    spec = parse_quantization("fp8_block")
    assert spec.for_node("LLM").format == "fp8_block"
    assert spec.for_node("vision_encoder").format == "fp8_block"


def test_rules_keep_their_order_then_keep_bf16_then_the_default():
    spec = parse_quantization({
        "format": "fp8_block",
        "layers": [
            {"match": "model.layers.0-1.**", "format": "bf16"},
            {"match": "model.layers.*.mlp.*", "format": "fp8_channel"},
        ],
        "keep_bf16": ["model.layers.*.self_attn.o_proj"],
    })
    node = spec.for_node("LLM")
    assert [r.format for r in node.rules] == [BF16, "fp8_channel", BF16]
    # first match wins: layer 1's MLP stays bf16, layer 5's takes the second rule
    assert node.rule_for("model.layers.1.mlp.gate_up_proj").format == BF16
    assert node.rule_for("model.layers.5.mlp.gate_up_proj").format == "fp8_channel"
    assert node.rule_for("model.layers.5.self_attn.o_proj").origin == "quantization.keep_bf16[0]"
    assert node.rule_for("model.layers.5.self_attn.qkv_proj") is None


def test_a_node_block_goes_before_the_top_level_and_inherits_its_default():
    spec = parse_quantization({
        "format": "fp8_block",
        "layers": [{"match": "lm_head", "format": "fp8_channel"}],
        "nodes": {"vision_encoder": {"keep_bf16": ["*"]}, "LLM": {"format": "fp8_tensor"}},
    })
    vision = spec.for_node("vision_encoder")
    assert vision.format == "fp8_block"
    assert [r.origin for r in vision.rules] == [
        "quantization.nodes.vision_encoder.keep_bf16[0]", "quantization.layers[0]",
    ]
    assert spec.for_node("LLM").format == "fp8_tensor"
    assert spec.for_node("Talker").format == "fp8_block"


@pytest.mark.parametrize("path, pattern, hit", [
    ("model.layers.3.mlp.gate_up_proj", "model.layers.*.mlp.*", True),
    ("model.layers.3.mlp.gate_up_proj", "model.layers.0-2.mlp.*", False),
    ("model.layers.2.mlp.gate_up_proj", "model.layers.0-2.mlp.*", True),
    # `*` is one segment, never a dot
    ("model.layers.3.mlp.gate_up_proj", "model.*", False),
    ("model.layers.3.mlp.gate_up_proj", "model.**", True),
    ("model.layers.3.mlp.gate_up_proj", "model.layers.2-4.**", True),
    ("model.layers.3.mlp.gate_up_proj", "**.gate_up_proj", True),
    ("model.layers.3.mlp.gate_up_proj", "**.mlp", False),
    ("model.layers.3.mlp.experts", r"re:model\.layers\.\d+\.mlp\.experts", True),
    ("model.layers.3.mlp.experts.x", r"re:model\.layers\.\d+\.mlp\.experts", False),
])
def test_patterns_name_module_paths(path, pattern, hit):
    assert pattern_matches(path, pattern) is hit
    assert Rule(pattern, BF16).matches(path) is hit


@pytest.mark.parametrize("raw, message", [
    ("fp9", "quantization.format must be one of"),
    ({"format": "fp8_block", "precision": "x"}, "unknown key(s) ['precision']"),
    ({"layers": [{"match": "re:(", "format": "bf16"}]}, "is not a valid regex"),
    ({"layers": [{"match": "lm_head"}]}, "exactly the keys ['match', 'format']"),
    ({"layers": [{"match": "lm_head", "format": "auto"}]}, "`auto` is only the default"),
    ({"layers": {"match": "lm_head", "format": "bf16"}}, "layers must be a list"),
    # a bare string would iterate as one-character patterns, and its `*` would match everything
    ({"keep_bf16": "lm_head"}, "keep_bf16 must be a list"),
    ({"keep_bf16": [""]}, "non-empty module-path pattern"),
    ({"kv_cache": "fp8_e4m3"}, "resources.<key>.quant"),
    ({"nodes": ["LLM"]}, "nodes must map node names"),
    ({"nodes": {"LLM": {"format": "int3"}}}, "quantization.nodes.LLM.format"),
    (["fp8_block"], "must be a format name or a mapping"),
])
def test_malformed_blocks_are_refused_with_the_key(raw, message):
    with pytest.raises(QuantizationSpecError) as info:
        parse_quantization(raw)
    assert any(message in problem for problem in info.value.problems), info.value.problems


def test_every_problem_is_reported_at_once():
    with pytest.raises(QuantizationSpecError) as info:
        parse_quantization({"format": "fp9", "keep_bf16": "x", "layers": [{"match": "re:(", "format": "bf16"}]})
    assert len(info.value.problems) == 3


def test_a_node_block_for_an_unplaced_node_is_refused():
    config = {"quantization": {"nodes": {"Thinker": {"format": "fp8_block"}}}}
    assert spec_problems(config, {"LLM"}) == [
        "quantization.nodes names ['Thinker'], which no node group places; placed: ['LLM']",
    ]
    assert spec_problems(config, {"Thinker"}) == []


def test_every_catalogued_format_is_nameable():
    for name in FORMATS:
        assert parse_quantization(name).top.format == name
