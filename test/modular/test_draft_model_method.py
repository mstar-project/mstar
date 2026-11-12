"""The ``draft_model`` and ``eagle3`` methods: the config's defaults and limits, a separate
draft model's declaration, and its refusals. A draft model one row behind its target (`DrafterNeeds.lag`)
is checked end to end on the GPU; these are the CPU halves."""
from __future__ import annotations

import json

import pytest
import torch

from mstar.engine.resources.draft.config import DraftConfig
from mstar.engine.resources.draft.drafter import DRAFT_KEEP_FIRST, ModelGeometry, VerifyNode, verify_writes
from mstar.engine.resources.draft.startup import geometry_problems
from mstar.engine.resources.step import Lag
from mstar.model.components.draft_model import DraftModel, draft_model_refusals
from mstar.model.components.qwen3_lm import Qwen3LMConfig

# Qwen3-0.6B's shapes, drafting for Qwen3-8B's
DRAFT = Qwen3LMConfig(hidden_size=1024, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                      head_dim=128, intermediate_size=3072, vocab_size=151936, tie_word_embeddings=True)
TARGET = ModelGeometry(num_layers=36, num_kv_heads=8, head_dim=128, hidden_size=4096, vocab_size=151936,
                       dtype="bfloat16")


def test_the_methods_default_and_bound_k():
    assert DraftConfig.parse({"method": "draft_model", "model": "m"}).num_draft_tokens == 3
    assert DraftConfig.parse({"method": "eagle3", "model": "m"}).num_draft_tokens == 3
    assert DraftConfig.parse({"method": "draft_model", "model": "m", "num_draft_tokens": 4}).num_draft_tokens == 4
    with pytest.raises(ValueError, match="at most 4 tokens"):
        DraftConfig.parse({"method": "draft_model", "model": "m", "num_draft_tokens": 7})
    with pytest.raises(ValueError, match="num_draft_tokens must be a positive integer"):
        DraftConfig.parse({"method": "dspark", "model": "m"})
    with pytest.raises(ValueError, match="set model"):
        DraftConfig.parse({"method": "draft_model"})


@pytest.mark.parametrize("k", [1, 3, 4])
def test_a_draft_model_lags_one_row_in_one_row_passes(k):
    decl = DraftModel(DRAFT, k, dtype="bfloat16", kv_key="kv_cache", pos_key="rope")
    needs = decl.needs
    assert needs.lag == Lag(1, (2,) + (1,) * (k - 1)) and sum(needs.lag.passes) == k + 1
    assert needs.kv.num_layers == 28 and (needs.kv.num_kv_heads, needs.kv.head_dim) == (8, 128)
    assert needs.carry.width == 1 and needs.carry.dtype == torch.int64
    assert DRAFT_KEEP_FIRST in verify_writes(needs)
    (binding,) = decl.node_resources({"LLM"})
    assert binding.config.layer_set == "draft_model" and binding.config.num_qo_heads == 16


def test_the_draft_model_non_goals_are_refused():
    # its depth is bounded by DraftConfig.parse alone (`test_the_methods_default_and_bound_k`)
    qwen3 = {"model_type": "qwen3", "vocab_size": 151936, "use_sliding_window": False}
    assert draft_model_refusals(qwen3, 151936) == []
    for hf, why in [
        ({**qwen3, "model_type": "qwen3_moe", "num_experts": 128}, "Qwen3 dense family"),
        ({**qwen3, "model_type": "llama"}, "Qwen3 dense family"),
        ({**qwen3, "vocab_size": 32000}, "another vocabulary"),
        ({**qwen3, "use_sliding_window": True}, "sliding-window"),
    ]:
        assert any(why in problem for problem in draft_model_refusals(hf, 151936)), (hf, why)


def _write_tokenizer(directory, words: list[str]) -> None:
    """A loadable tokenizer of ``words``: the target's own is loaded at construction (its stop
    ids), before the draft model's is compared."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token=words[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))


def test_a_draft_model_with_another_tokenizer_is_refused(tmp_path):
    pytest.importorskip("tokenizers")
    from qwen3_cpu import TARGET, make_model, write_checkpoints

    target, _ = write_checkpoints(tmp_path)
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(TARGET))
    _write_tokenizer(target, ["<unk>", "a", "b"])
    _write_tokenizer(draft, ["<unk>", "c", "d"])
    block = {"method": "draft_model", "model": str(draft), "num_draft_tokens": 3}
    with pytest.raises(ValueError, match="another tokenizer"):
        make_model(target, block)
    _write_tokenizer(draft, ["<unk>", "a", "b"])
    assert make_model(target, block).drafter_decl is not None


def test_startup_takes_a_draft_models_own_width_and_still_its_vocabulary():
    """A draft model reads only tokens, so its residual width is its own; its vocabulary must be the
    target's (no token map for a draft model)."""
    decl = DraftModel(DRAFT, 3, dtype="bfloat16", kv_key="kv_cache", pos_key="rope")
    node = VerifyNode("LLM", "decode", DraftConfig("draft_model", 3, "m"), decl.needs, target=TARGET)
    assert not [p for p in geometry_problems(node, {}) if "residual width" in p]
    other = DraftModel(Qwen3LMConfig(**{**DRAFT.__dict__, "vocab_size": 32000}), 3, dtype="bfloat16",
                       kv_key="kv_cache", pos_key="rope")
    node = VerifyNode("LLM", "decode", DraftConfig("draft_model", 3, "m"), other.needs, target=TARGET)
    assert any("vocabulary" in p for p in geometry_problems(node, {}))


def test_the_built_node_drafts_with_its_draft_model_module(tmp_path):
    """The draft model's drafter is an nn.Module: it lives in the node's ``_modules``, and nothing at
    class level may shadow it (the engine refuses a verifying node whose ``drafter`` reads None)."""
    from qwen3_cpu import TARGET, bind_share, build_submodule, make_model, write_checkpoints

    from mstar.model.components.draft_model import DraftModelDrafter

    target, _ = write_checkpoints(tmp_path)
    draft = tmp_path / "draft"
    draft.mkdir()
    # the CPU target's family and vocabulary, two layers
    (draft / "config.json").write_text(json.dumps({**TARGET, "num_hidden_layers": 2}))
    model = make_model(target, {"method": "draft_model", "model": str(draft), "num_draft_tokens": 3}, max_seq_len=512)
    sub = build_submodule(model)
    assert isinstance(sub.drafter, DraftModelDrafter) and sub.drafter is sub._modules["drafter"]
    # k is the engine's to bind at load
    assert bind_share(sub, 3)._block == 4
