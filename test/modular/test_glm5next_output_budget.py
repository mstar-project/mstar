"""GLM-5.3-Flash's output budget, decode Loop cap and chat-template call.

check_stop enforces the per-request ``max_tokens``; the Loop's ``max_iters``
is only the runaway guard. It used to be the default budget (1024), so a
request asking for more was cut at 1025 tokens with no error.
"""

import torch

from mstar.graph.base import Loop
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel


def _model(**kwargs) -> Glm5NextModel:
    # byte tokenizer: no HF IO at construction
    return Glm5NextModel("x", tokenizer_mode="byte", **kwargs)


def _decode(m: Glm5NextModel) -> Loop:
    decode = m.get_graph_walk_graphs()["decode"]
    assert isinstance(decode, Loop)
    return decode


def test_request_budget_above_the_default_is_honored():
    m = _model()
    assert m.get_max_output_tokens() == m.config.max_output_tokens == 1024
    assert m.get_max_output_tokens(max_output_tokens=2000) == 2000
    # each decode iteration emits a token on top of the prefill's
    assert _decode(m).max_iters >= 2000 - 1


def test_budget_and_loop_cap_are_held_to_the_window():
    m = _model()
    topk = m.config.index_topk
    assert m.get_max_output_tokens(max_output_tokens=10_000) == topk
    # a one-token prompt's context reaches index_topk after topk - 1 iterations
    assert _decode(m).max_iters == topk - 1


def test_mtp_loop_cap_is_held_to_the_cache_rows():
    m = _model(mtp_num_draft_tokens=3)
    cfg = m.config
    # every verify step stores k + 1 cache rows, accepted or not
    assert _decode(m).max_iters == min(cfg.index_topk - 1, (cfg.kv_rows - 1) // 4)


def test_chat_template_returns_the_bare_ids():
    class ChatTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, **kwargs):
            # transformers 5 returns a BatchEncoding unless return_dict=False
            assert kwargs.get("return_dict") is False
            return torch.tensor([[7, 8, 9]])

    m = Glm5NextModel("x")
    m._tokenizer = ChatTokenizer()
    ids = m.process_prompt("hi", ["text"], ["text"])["text_inputs"][0]
    assert ids.tolist() == [7, 8, 9]


def test_long_context_budget_follows_the_window():
    # held to index_topk, a long-context request was cut at 2048 tokens
    m = _model(dsa_long_context=True, max_seq_len=1 << 20)
    assert m.get_max_output_tokens(max_output_tokens=5000) == 5000
    assert _decode(m).max_iters == m.config.context_limit - 1 == (1 << 20) - 1
