"""Orpheus LM weight loading against a tiny HF ``LlamaForCausalLM`` state dict.

``canopylabs/orpheus-3b-0.1-ft`` ties embeddings; a copy written by
``save_pretrained`` stores no ``lm_head.weight``, so the head must come from
the embedding, and a parameter no checkpoint key fills must fail the load.
"""

import pytest
import torch

pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.model.orpheus.components.language_model import OrpheusForCausalLM  # noqa: E402
from mstar.model.orpheus.config import OrpheusModelConfig  # noqa: E402

ROPE_SCALING = {
    "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
    "original_max_position_embeddings": 64, "rope_type": "llama3",
}


def _config(tie: bool = True) -> OrpheusModelConfig:
    return OrpheusModelConfig(
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=32, head_dim=8,
        intermediate_size=48, max_position_embeddings=128, vocab_size=96, rope_scaling=dict(ROPE_SCALING),
        pad_token_id=95, **({} if tie else {"tie_word_embeddings": False}),
    )


def _hf_state(tie: bool = True) -> dict[str, torch.Tensor]:
    cfg = _config(tie)
    torch.manual_seed(0)
    hf = LlamaForCausalLM(LlamaConfig(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta, rope_scaling=dict(ROPE_SCALING),
        max_position_embeddings=cfg.max_position_embeddings, tie_word_embeddings=tie,
    ))
    return {k: v.detach().clone() for k, v in hf.state_dict().items()}


def _ours(tie: bool = True, comm_group: CommGroup | None = None) -> OrpheusForCausalLM:
    lm = OrpheusForCausalLM(_config(tie), comm_group=comm_group or CommGroup.trivial())
    for p in lm.parameters():
        p.data.zero_()
    return lm


def _without(state: dict[str, torch.Tensor], fragment: str) -> list[tuple[str, torch.Tensor]]:
    return [(k, v) for k, v in state.items() if fragment not in k]


def test_head_is_tied_when_the_checkpoint_stores_none():
    state = _hf_state()
    lm = _ours()
    loaded = lm.load_weights(_without(state, "lm_head"))
    assert "lm_head.weight" in loaded
    assert torch.equal(lm.lm_head.weight, state["model.embed_tokens.weight"])


def test_a_stored_head_is_kept():
    state = _hf_state(tie=False)
    lm = _ours(tie=False)
    lm.load_weights(list(state.items()))
    assert torch.equal(lm.lm_head.weight, state["lm_head.weight"])
    assert not torch.equal(lm.lm_head.weight, lm.model.embed_tokens.weight)


def test_tp_rank_ties_its_own_vocab_shard():
    state = _hf_state()
    lm = _ours(comm_group=CommGroup(my_global_rank=1, my_group_rank=1, group_members=[0, 1]))
    lm.load_weights(_without(state, "lm_head"))
    assert torch.equal(lm.lm_head.weight, state["model.embed_tokens.weight"][48:])


def test_untied_checkpoint_without_head_raises():
    with pytest.raises(RuntimeError, match="does not tie"):
        _ours(tie=False).load_weights(_without(_hf_state(tie=False), "lm_head"))


def test_unloaded_parameter_raises():
    with pytest.raises(RuntimeError, match="unloaded"):
        _ours().load_weights(_without(_hf_state(), "layers.1.mlp"))
