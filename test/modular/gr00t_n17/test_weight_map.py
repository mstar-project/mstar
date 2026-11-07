import json
import re

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.weight_loader import NODE_PREFIXES, SKIPPED, _remapper, load_gr00t_n17_weights

from .helpers import BACKBONE, GR00T, needs_checkpoint


class _StateEncoderNode(nn.Module):
    """A node holds its parts under the reference's names (``state_encoder.layer1.W``)."""

    def __init__(self):
        super().__init__()
        layer1 = nn.ParameterDict({"W": nn.Parameter(torch.zeros(2, 3, 4)), "b": nn.Parameter(torch.zeros(2, 4))})
        self.state_encoder = nn.ModuleDict({"layer1": layer1})


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
    """Shapes the config implies for one LLM tensor of each kind (``N`` = any index)."""
    t = c.text
    return {
        "backbone.model.model.language_model.embed_tokens.weight": (t.vocab_size, t.hidden_size),
        "backbone.model.model.language_model.layers.N.self_attn.q_proj.weight":
            (t.num_attention_heads * t.head_dim, t.hidden_size),
        "backbone.model.model.language_model.layers.N.self_attn.k_proj.weight":
            (t.num_key_value_heads * t.head_dim, t.hidden_size),
        "backbone.model.model.language_model.layers.N.self_attn.q_norm.weight": (t.head_dim,),
        "backbone.model.model.language_model.layers.N.mlp.gate_proj.weight": (t.intermediate_size, t.hidden_size),
    }


@needs_checkpoint
def test_the_checkpoint_matches_the_config():
    c = Gr00tN17Config.from_hf(GR00T, BACKBONE)
    weight_map = json.loads((GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    # the load drops a tensor no node claims without a word, so this is what catches one
    assert {k for k in weight_map if not any(_remapper(node)(k) for node in NODE_PREFIXES)} == SKIPPED

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
