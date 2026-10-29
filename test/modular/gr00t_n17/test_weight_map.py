import json
import re

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.weight_loader import SKIPPED, load_gr00t_n17_weights, node_of

from .helpers import BACKBONE, GR00T, needs_checkpoint


@pytest.mark.parametrize("name, node", [
    ("backbone.model.model.visual.blocks.3.attn.qkv.weight", "vision_encoder"),
    ("backbone.model.model.visual.deepstack_merger_list.1.norm.weight", "vision_encoder"),
    ("backbone.model.model.language_model.layers.15.self_attn.q_proj.weight", "LLM"),
    ("backbone.model.model.language_model.embed_tokens.weight", "LLM"),
    ("action_head.vlln.weight", "vl_bridge"),
    ("action_head.vl_self_attention.transformer_blocks.0.attn1.to_q.weight", "vl_bridge"),
    ("action_head.state_encoder.layer1.W", "state_encoder"),
    ("action_head.model.transformer_blocks.31.norm1.linear.weight", "denoise_step"),
    ("action_head.position_embedding.weight", "denoise_step"),
    ("backbone.model.model.language_model.norm.weight", None),
    ("backbone.model.lm_head.weight", None),
])
def test_routes_each_tensor_to_its_node(name, node):
    assert node_of(name) == node


def test_an_unclaimed_tensor_fails_loudly():
    with pytest.raises(ValueError, match="no GR00T node"):
        node_of("action_head.something_new.weight")


class _StateEncoderNode(nn.Module):
    """A node holds its parts under the reference's names (``state_encoder.layer1.W``)."""

    def __init__(self, extra: bool = False):
        super().__init__()
        layer1 = nn.ParameterDict({"W": nn.Parameter(torch.zeros(2, 3, 4)), "b": nn.Parameter(torch.zeros(2, 4))})
        self.state_encoder = nn.ModuleDict({"layer1": layer1})
        if extra:
            self.unloaded = nn.Parameter(torch.zeros(1))


def test_a_node_loads_only_its_own_tensors_and_all_of_them(tmp_path):
    w = torch.arange(24.0).reshape(2, 3, 4)
    save_file({
        "action_head.state_encoder.layer1.W": w,
        "action_head.state_encoder.layer1.b": torch.ones(2, 4),
        "action_head.action_decoder.layer1.W": torch.ones(2, 3, 4),  # another node's
    }, tmp_path / "model.safetensors")
    module = _StateEncoderNode()
    loaded = load_gr00t_n17_weights(module, "state_encoder", tmp_path)
    assert loaded == {"state_encoder.layer1.W", "state_encoder.layer1.b"}
    assert torch.equal(module.state_encoder["layer1"]["W"], w)
    with pytest.raises(ValueError, match="got no weight"):
        load_gr00t_n17_weights(_StateEncoderNode(extra=True), "state_encoder", tmp_path)


def test_a_tensor_under_the_nodes_prefix_that_lands_nowhere_is_refused(tmp_path):
    """A renamed module would load everything else and serve that part at its init."""
    save_file({
        "action_head.state_encoder.layer1.W": torch.zeros(2, 3, 4),
        "action_head.state_encoder.layer1.b": torch.zeros(2, 4),
        "action_head.state_encoder.layer2.W": torch.zeros(2, 4, 5),
    }, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match=r"1 tensor\(s\) matched no parameter.*layer2\.W"):
        load_gr00t_n17_weights(_StateEncoderNode(), "state_encoder", tmp_path)


def _expected_shapes(c: Gr00tN17Config) -> dict[str, tuple[int, ...]]:
    """Shapes the config implies for one tensor of each kind (``N`` = any index)."""
    t, v, dit, br = c.text, c.vision, c.dit, c.bridge
    merged = v.hidden_size * v.merge_unit
    e, d = c.max_num_embodiments, c.input_embedding_dim
    return {
        "backbone.model.model.visual.patch_embed.proj.weight":
            (v.hidden_size, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size),
        "backbone.model.model.visual.pos_embed.weight": (v.num_position_embeddings, v.hidden_size),
        "backbone.model.model.visual.blocks.N.attn.qkv.weight": (3 * v.hidden_size, v.hidden_size),
        "backbone.model.model.visual.blocks.N.mlp.linear_fc1.weight": (v.intermediate_size, v.hidden_size),
        "backbone.model.model.visual.merger.norm.weight": (v.hidden_size,),
        "backbone.model.model.visual.merger.linear_fc2.weight": (v.out_hidden_size, merged),
        # DeepStack mergers normalise after the 2x2 merge, hence the wider norm
        "backbone.model.model.visual.deepstack_merger_list.N.norm.weight": (merged,),
        "backbone.model.model.visual.deepstack_merger_list.N.linear_fc2.weight": (v.out_hidden_size, merged),
        "backbone.model.model.language_model.embed_tokens.weight": (t.vocab_size, t.hidden_size),
        "backbone.model.model.language_model.layers.N.self_attn.q_proj.weight":
            (t.num_attention_heads * t.head_dim, t.hidden_size),
        "backbone.model.model.language_model.layers.N.self_attn.k_proj.weight":
            (t.num_key_value_heads * t.head_dim, t.hidden_size),
        "backbone.model.model.language_model.layers.N.self_attn.q_norm.weight": (t.head_dim,),
        "backbone.model.model.language_model.layers.N.mlp.gate_proj.weight": (t.intermediate_size, t.hidden_size),
        "action_head.vlln.weight": (c.backbone_embedding_dim,),
        "action_head.vl_self_attention.transformer_blocks.N.attn1.to_q.weight": (br.inner_dim, br.inner_dim),
        "action_head.state_encoder.layer1.W": (e, c.max_state_dim * c.state_history_length, c.hidden_size),
        "action_head.state_encoder.layer2.W": (e, c.hidden_size, d),
        "action_head.action_encoder.W1.W": (e, c.max_action_dim, d),
        "action_head.action_encoder.W2.W": (e, 2 * d, d),
        "action_head.action_decoder.layer2.W": (e, c.hidden_size, c.max_action_dim),
        "action_head.position_embedding.weight": (c.max_seq_len, d),
        "action_head.model.transformer_blocks.N.attn1.to_q.weight": (dit.inner_dim, dit.inner_dim),
        "action_head.model.transformer_blocks.N.norm1.linear.weight": (2 * dit.inner_dim, dit.inner_dim),
        "action_head.model.proj_out_2.weight": (dit.output_dim, dit.inner_dim),
    }


@needs_checkpoint
def test_the_checkpoint_matches_the_config():
    c = Gr00tN17Config.from_hf(GR00T, BACKBONE)
    weight_map = json.loads((GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    nodes = {name: node_of(name) for name in weight_map}  # raises on an unclaimed tensor
    assert {n for n, node in nodes.items() if node is None} == SKIPPED

    shapes = {}
    for shard in set(weight_map.values()):
        with safe_open(GR00T / shard, "pt") as f:  # headers only, no tensor data
            shapes |= {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    expected = _expected_shapes(c)
    for pattern, shape in expected.items():
        hits = [k for k in shapes if re.fullmatch(re.escape(pattern).replace("N", r"\d+"), k)]
        assert hits, pattern
        assert {shapes[k] for k in hits} == {shape}, pattern

    def count(prefix):
        return len({k.split(".")[len(prefix.split("."))] for k in shapes if k.startswith(prefix + ".")})

    assert count("backbone.model.model.language_model.layers") == c.text.num_hidden_layers
    assert count("backbone.model.model.visual.blocks") == c.vision.depth
    assert count("backbone.model.model.visual.deepstack_merger_list") == len(c.deepstack_visual_indexes)
    assert count("action_head.model.transformer_blocks") == c.dit.num_layers
    assert count("action_head.vl_self_attention.transformer_blocks") == c.bridge.num_layers
    # even DT blocks cross-attend the backbone's features, odd ones self-attend
    k = "action_head.model.transformer_blocks.{}.attn1.to_k.weight"
    assert shapes[k.format(0)] == (c.dit.inner_dim, c.backbone_embedding_dim)
    assert shapes[k.format(1)] == (c.dit.inner_dim, c.dit.inner_dim)
