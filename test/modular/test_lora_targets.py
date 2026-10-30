"""What the LoRA checkpoint reader parses and refuses, by name, and the embedding's gather op,
on CPU.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

pytest.importorskip("safetensors")
from lora_module_helpers import rand  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mstar.engine.resources.lora.checkpoint import normalize_lora_key, parse_lora_key, read_lora_file  # noqa: E402
from mstar.engine.resources.lora.ops import lora_gather_shrink  # noqa: E402


@pytest.mark.parametrize("key,want", [
    ("base_model.model.model.embed_tokens.lora_embedding_A", ("model.embed_tokens", "A")),
    ("base_model.model.layers.0.q_proj.lora_down.weight", ("layers.0.q_proj", "A")),
    ("base_model.model.layers.0.q_proj.alpha", ("layers.0.q_proj", "alpha")),
    ("base_model.model.layers.0.q_proj.lora_magnitude_vector", "DoRA"),
    ("lora_unet_double_blocks_0_img_attn_qkv.dora_scale", "DoRA"),
    ("base_model.model.lm_head.modules_to_save.weight", "modules_to_save"),
    ("base_model.model.embed_tokens.token_adapter.trainable_tokens_delta", "trainable_token"),
    ("base_model.model.layers.0.q_proj.lora_B.bias", "lora_bias"),
    ("base_model.model.score.weight", "not a lora_A"),
    ("base_model.model.layers.0.q_proj.oft_R.weight", "not a lora_A"),
])
def test_a_checkpoint_key_is_parsed_or_refused_by_what_it_is(key, want):
    if isinstance(want, str):
        with pytest.raises(ValueError, match=want):
            parse_lora_key(normalize_lora_key(key))
    else:
        assert parse_lora_key(normalize_lora_key(key)) == want


PATTERN_RANKS = {(0, "q_proj"): 4, (0, "k_proj"): 2, (0, "v_proj"): 4, (1, "v_proj"): 4}


@pytest.mark.parametrize("config,want", [
    # alpha / r per module: k at its own rank, v on layer 0 by its own alpha
    ({"r": 4, "lora_alpha": 8, "rank_pattern": {"k_proj": 2}, "alpha_pattern": {"layers.0.self_attn.v_proj": 32}},
     {("0", "q_proj"): 8, ("0", "k_proj"): 8, ("0", "v_proj"): 32, ("1", "v_proj"): 8}),
    # rslora: alpha / sqrt(r), stored as the alpha' whose alpha' / r equals it
    ({"r": 4, "lora_alpha": 8, "use_rslora": True, "alpha_pattern": {"v_proj": 16}},
     {("0", "q_proj"): 16, ("0", "k_proj"): 8 * math.sqrt(2), ("0", "v_proj"): 32, ("1", "v_proj"): 32}),
    # an alpha_pattern alone: the rest take alpha = r
    ({"alpha_pattern": {"q_proj": 12}},
     {("0", "q_proj"): 12, ("0", "k_proj"): 2, ("0", "v_proj"): 4, ("1", "v_proj"): 4}),
    # control, no patterns: lora_alpha everywhere (wrong on the modules patterns name)
    ({"r": 4, "lora_alpha": 8}, {("0", "q_proj"): 8, ("0", "k_proj"): 8, ("0", "v_proj"): 8, ("1", "v_proj"): 8}),
    # a rank_pattern the weights contradict
    ({"r": 4, "lora_alpha": 8, "rank_pattern": {"k_proj": 8}}, "rank_pattern"),
])
def test_alpha_and_rank_patterns_scale_each_module_as_peft_does(config, want, tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for (layer, proj), r in PATTERN_RANKS.items():
        module = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        tensors[f"{module}.lora_A.weight"] = torch.randn(r, 8, generator=g)
        tensors[f"{module}.lora_B.weight"] = torch.randn(8, r, generator=g)
    path = str(tmp_path / "adapter_model.safetensors")
    save_file(tensors, path)
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    if isinstance(want, str):
        with pytest.raises(ValueError, match=want):
            read_lora_file(path)
        return
    alphas = {tuple(m.split(".")[-3::2]): layer.alpha for m, layer in read_lora_file(path).layers.items()}
    assert alphas == pytest.approx(want)


def test_gather_shrink_op_matches_its_fake():
    torch.library.opcheck(
        lora_gather_shrink,
        # tokens on slots [1, 2, 0, 1], grouped by slot
        (torch.tensor([0, 5, -1, 31]), rand(3, 4, 32, seed=1), torch.tensor([2, 0, 3, 1], dtype=torch.int32),
         torch.tensor([0, 1, 3, 4], dtype=torch.int32), torch.tensor([0, 4, 2], dtype=torch.int32)),
    )
